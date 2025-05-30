// const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
// const { OllamaEmbeddings } = require("@langchain/community/embeddings/ollama");
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
// const { HumanMessage, AIMessage } = require("@langchain/core/messages");
import { HumanMessage, AIMessage } from "@langchain/core/messages";
// const { RunnableSequence } = require("@langchain/core/runnables");
import { RunnableSequence } from "@langchain/core/runnables";

// Assuming you have a function to run Neo4j queries
// const { runQuery } = require("./index.js");
import { runQuery } from "./index.js";

// Assuming you have a chat history management system
const mainChatMessageHistory = require("./index.js");

// Decider: Determines if the question is related to Bangladeshi law and rewrites it if necessary
const decider = async (queryText) => {
    const model = new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        temperature: 0,
        maxRetries: 2,
        apiKey: process.env.GOOGLE_API_KEY,
    });

    const prompt = `You are a law assistant for question-answering tasks on Bangladeshi legislature.
        Determine if the following question is related to Bangladeshi law. If it is, rewrite the question to optimize it for semantic retrieval. If it is not, respond with "This question is not related to Bangladeshi law."

        Question: "${queryText}"

        Response:`;

    const response = await model.invoke(prompt);
    return response.content;
};

// Retriever: Queries Neo4j for relevant law chunks
const retriever = async (queryText) => {
    const ollamaEmbedder = new OllamaEmbeddings({ model: "bge-m3:latest" });
    const queryVector = await ollamaEmbedder.embedQuery(queryText);

    const neo4jQuery = `
        CALL db.index.vector.queryNodes('text_chunks', 5, $vector)
        YIELD node, score
        OPTIONAL MATCH path = (node)-[:NEXT_CHUNK*0..10]->(next_chunk)
        WITH REDUCE(
          collectedChunks = [node], 
          c IN NODES(path) | 
          CASE WHEN c IN collectedChunks THEN collectedChunks ELSE collectedChunks + c END
        ) AS unique_chunks, score
        UNWIND unique_chunks AS chunk
        RETURN DISTINCT chunk, score
        ORDER BY chunk.chunk_index
    `;

    const records = await runQuery(neo4jQuery, { vector: queryVector });

    const retrievedDocs = records
        .map((record) => {
            const node = record.get("chunk");
            const chunkText = node.properties.chunk_text || "";
            const lawTitle = node.properties.law_title || "";
            const sectionNumber = node.properties.section_number || "";

            return `[${lawTitle} | Section ${sectionNumber}] ${chunkText}`;
        })
        .join("\n");

    return retrievedDocs;
};

// Answerer: Generates a response based on the retrieved context
const answerer = async (queryText, retrievedDocs) => {
    const model = new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        temperature: 0,
        maxRetries: 2,
        apiKey: process.env.GOOGLE_API_KEY,
    });

    const prompt = `You are a law assistant for question-answering tasks on Bangladeshi legislature.
        Use the following pieces of retrieved context to answer the question.
        Don't start your answer with "Based on the retrieved context, ...". And if the question is a greeting or something irrelevant, answer accordingly.
        If you don't know the answer, just say that you don't know.
        Question: "${queryText}"\n\nContext: "${retrievedDocs}\n\nAnswer:"
        Answer:`;

    const response = await model.invoke(prompt);
    return response.content;
};

// Chain: Combines decider, retriever, and answerer
export default chain = RunnableSequence.from([
    {
        queryText: (input) => input.queryText,
    },
    {
        decision: async (input) => await decider(input.queryText),
        queryText: (input) => input.queryText,
    },
    {
        retrievedDocs: async (input) => {
            if (input.decision === "This question is not related to Bangladeshi law.") {
                return null;
            }
            return await retriever(input.decision);
        },
        queryText: (input) => input.queryText,
        decision: (input) => input.decision,
    },
    {
        answer: async (input) => {
            if (input.decision === "This question is not related to Bangladeshi law.") {
                return input.decision;
            }
            return await answerer(input.queryText, input.retrievedDocs);
        },
    },
]);

