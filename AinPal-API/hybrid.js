const express = require('express');
const neo4j = require('neo4j-driver');
const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { OllamaEmbeddings } = require('@langchain/ollama');
const { ChatMessageHistory } = require("langchain/stores/message/in_memory");
const { HumanMessage, AIMessage } = require("@langchain/core/messages");
import { ConversationChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";

require('dotenv').config();
const app = express();
app.use(express.json());
const port = 8000;

const neo4jDriver = neo4j.driver(
  'bolt://localhost:7999',
  neo4j.auth.basic(process.env.NEO4J_USERNAME, process.env.NEO4J_PASSWORD)
);

const runQuery = async (query, params = {}) => {
    const session = neo4jDriver.session();
    try {
        const result = await session.run(query, params);
        return result.records;
    } finally {
        await session.close();
    }
};

// Create a global BufferMemory instance.
// In a multi-user production setup, you would create one per session/user.
const memory = new BufferMemory();

// The /hybrid-chat endpoint now includes a ConversationChain using the above memory.
app.post("/hybrid-chat", async (req, res) => {
    const { queryText } = req.body;

    if (!queryText) {
        return res.status(400).json({ error: "Query text is required" });
    }

    try {
        // Initialize embeddings and model
        const ollamaEmbedder = new OllamaEmbeddings({ model: "bge-m3:latest" });
        const model = new ChatGoogleGenerativeAI({
            model: "gemini-1.5-flash",
            temperature: 0,
            apiKey: process.env.GOOGLE_API_KEY,
        });

        // Step 1: Hybrid search – vector search and keyword search
        const queryVector = await ollamaEmbedder.embedQuery(queryText);
        const vectorSearchQuery = `
            CALL db.index.vector.queryNodes('text_chunks', 8, $vector)
            YIELD node, score
            RETURN node, score
        `;
        const vectorResults = await runQuery(vectorSearchQuery, { vector: queryVector });

        const keywords = queryText.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 3);
        let keywordSearchQuery = '';
        let keywordParams = {};

        if (keywords.length > 0) {
            keywordSearchQuery = `
                MATCH (chunk:TextChunk)
                WHERE ${keywords.map((_, i) => `toLower(chunk.chunk_text) CONTAINS $keyword${i}`).join(' OR ')}
                RETURN chunk as node, 0.5 as score
                LIMIT 8
            `;
            keywords.forEach((keyword, i) => {
                keywordParams[`keyword${i}`] = keyword;
            });
        }
        const keywordResults = keywords.length > 0 ? 
            await runQuery(keywordSearchQuery, keywordParams) : [];

        const allResults = [...vectorResults, ...keywordResults];
        const uniqueNodes = new Map();
        allResults.forEach(record => {
            const node = record.get("node");
            const score = record.get("score");
            const nodeId = node.identity.toString();
            if (!uniqueNodes.has(nodeId) || uniqueNodes.get(nodeId).score < score) {
                uniqueNodes.set(nodeId, { node, score });
            }
        });
        const combinedResults = Array.from(uniqueNodes.values())
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);

        // Expand the results by retrieving adjacent chunks
        const expandedNodes = new Set();
        const expandedResults = [];
        for (const result of combinedResults) {
            const nodeId = result.node.identity.toString();
            if (!expandedNodes.has(nodeId)) {
                expandedNodes.add(nodeId);
                expandedResults.push(result);
                // Fetch adjacent chunks
                const expandQuery = `
                    MATCH (node)-[:NEXT_CHUNK*1..2]->(next_chunk)
                    WHERE id(node) = $nodeId
                    RETURN next_chunk as node, 0.4 as score
                    LIMIT 3
                `;
                const expandedChunks = await runQuery(expandQuery, { nodeId: result.node.identity });
                for (const expandedRecord of expandedChunks) {
                    const expandedNode = expandedRecord.get("node");
                    const expandedNodeId = expandedNode.identity.toString();
                    if (!expandedNodes.has(expandedNodeId)) {
                        expandedNodes.add(expandedNodeId);
                        expandedResults.push({ 
                            node: expandedNode, 
                            score: expandedRecord.get("score") 
                        });
                    }
                }
            }
        }
        const retrievedDocs = expandedResults
            .map((result) => {
                const node = result.node;
                const chunkText = node.properties.chunk_text || "";
                const lawTitle = node.properties.law_title || "";
                const sectionNumber = node.properties.section_number || "";
                return `\n\n[${lawTitle} | Section ${sectionNumber}]\n ${chunkText}`;
            })
            .join("\n");

        // Logging for debugging
        expandedResults.forEach((result) => {
            const node = result.node;
            const chunkText = node.properties.chunk_text || "";
            const lawTitle = node.properties.law_title || "";
            const sectionNumber = node.properties.section_number || "";
            const score = result.score;
            console.log(`[Score: ${score.toFixed(3)}] [${lawTitle} | Section ${sectionNumber}] ${chunkText}`);
            console.log("-------------------------------------");
        });

        // Construct the prompt with retrieved context.
        // Here, we include our chat history via the ConversationChain's memory.
        const basePrompt = `You are a law assistant for question-answering tasks on Bangladeshi legislature.
Use the following pieces of retrieved context to answer the question.
Don't start your answer with "Based on the retrieved context, ...". And if the question is a greeting or something irrelevant, answer accordingly.
If you don't know the answer, just say that you don't know. Don't say something like "The retrieved context is not enough to answer the question."
Question: "${queryText}"
Context: "${retrievedDocs}"
Answer:`;

        // Create a ConversationChain instance using the current model and the global BufferMemory.
        const conversationChain = new ConversationChain({
            llm: model,
            memory: memory,
        });

        // Call the conversation chain with the prompt.
        // The chain will automatically add prior conversation history from the memory.
        const result = await conversationChain.call({ input: basePrompt });

        res.status(200).json({
            answer: result.response,
            debug: {
                retrievedDocCount: expandedResults.length,
                vectorResultsCount: vectorResults.length,
                keywordResultsCount: keywordResults.length,
                combinedResultsCount: combinedResults.length
            }
        });
    
    } catch (error) {
        console.error("Error handling /hybrid-chat request:", error);
        res.status(500).json({ error: "An error occurred while processing the request" });
    }
});

// ... other routes remain unchanged ...

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
// Close the Neo4j driver when the application is terminated