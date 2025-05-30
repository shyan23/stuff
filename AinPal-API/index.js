const express = require("express");
const neo4j = require("neo4j-driver");
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { OllamaEmbeddings } = require("@langchain/ollama");
const { ChatMessageHistory } = require("langchain/stores/message/in_memory");
const { HumanMessage, AIMessage } = require("@langchain/core/messages");
const { BufferMemory } = require("langchain/memory");
const start = process.hrtime(); // High-resolution time start
const startMem = process.memoryUsage().heapUsed; // Memory before handling


require("dotenv").config();
const app = express();
app.use(express.json());
const port = 8000;
const neo4jDriver = neo4j.driver(
  "bolt://localhost:7999",
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

// Create a map to store chat histories for different sessions
const chatHistories = new Map();

// app.post("/chat", async (req, res) => {
//   const { queryText, sessionId = "default" } = req.body;
//   if (!queryText) {
//     return res.status(400).json({ error: "Query text is required" });
//   }
//   try {
//     // Get or create message history for this session
//     if (!chatHistories.has(sessionId)) {
//       chatHistories.set(sessionId, new ChatMessageHistory());
//     }
//     const messageHistory = chatHistories.get(sessionId);

//     // Step 1: Query Neo4j for retrieval
//     ollamaEmbedder = new OllamaEmbeddings({ model: "bge-m3:latest" });
//     const model = new ChatGoogleGenerativeAI({
//       model: "gemini-1.5-flash",
//       temperature: 0,
//       apiKey: process.env.GOOGLE_API_KEY,
//     });
//     const queryVector = await ollamaEmbedder.embedQuery(queryText);
//     const neo4jQuery = `
//  CALL db.index.vector.queryNodes('text_chunks', 5, $vector)
//  YIELD node, score
//  OPTIONAL MATCH path = (node)-[:NEXT_CHUNK*0..10]->(next_chunk)
//  WITH REDUCE(
//  collectedChunks = [node],
//  c IN NODES(path) |
//  CASE WHEN c IN collectedChunks THEN collectedChunks ELSE collectedChunks + c END
//  ) AS unique_chunks, score
//  UNWIND unique_chunks AS chunk
//  RETURN DISTINCT chunk, score
//  ORDER BY chunk.chunk_index
//  `;
//     const records = await runQuery(neo4jQuery, { vector: queryVector });
//     // Build up a string that includes chunk_text + law_title + section_number
//     const retrievedDocs = records
//       .map((record) => {
//         const node = record.get("chunk");
//         const chunkText = node.properties.chunk_text || "";
//         const lawTitle = node.properties.law_title || "";
//         const sectionNumber = node.properties.section_number || "";
//         // Attach both law_title and section_number with the chunk text
//         return `\n\n[${lawTitle} | Section ${sectionNumber}]\n ${chunkText}`;
//       })
//       .join("\n");
//     // Optionally print each record for debugging/logging
//     records.forEach((record) => {
//       const node = record.get("chunk");
//       const chunkText = node.properties.chunk_text || "";
//       const lawTitle = node.properties.law_title || "";
//       const sectionNumber = node.properties.section_number || "";
//       console.log(`[${lawTitle} | Section ${sectionNumber}] ${chunkText}`);
//       console.log("-------------------------------------");
//     });

//     // Get chat history
//     const pastMessages = await messageHistory.getMessages();
//     const chatHistory = pastMessages
//       .map((msg) => {
//         if (msg._getType() === "human") {
//           return `Human: ${msg.content}`;
//         } else {
//           return `AI: ${msg.content}`;
//         }
//       })
//       .join("\n");

//     // Add the current message to history
//     await messageHistory.addUserMessage(queryText);

//     const prompt = `You are a law assistant for question-answering tasks on Bangladeshi legislature.
//  Use the following pieces of retrieved context to answer the question.
//  Don't start your answer with "Based on the retrieved context, ...". And if the question is a greeting or something irrelevant, answer accordingly.
//  If you don't know the answer, just say that you don't know. Don't say something like "The retrieved context is not enough to answer the question."
 
//  Chat history:
//  ${chatHistory}
 
//  Question: "${queryText}"\n\nContext: "${retrievedDocs}\n\nAnswer:"
//  Answer:`;
//     const response = await model.invoke(prompt);

//     // Add AI response to history
//     await messageHistory.addAIMessage(response.content);

//     res.status(200).json({ answer: response.content });
//   } catch (error) {
//     console.error("Error handling /chat request:", error);
//     res
//       .status(500)
//       .json({ error: "An error occurred while processing the request" });
//   }
// });

// Apply the same memory pattern to hybrid-chat endpoint
app.post("/hybrid-chat", async (req, res) => {
  const startTime = process.hrtime();
  const startMemory = process.memoryUsage().heapUsed;

  const { queryText, sessionId = "default" } = req.body;
  if (!queryText) {
    return res.status(400).json({ error: "Query text is required" });
  }

  try {
    if (!chatHistories.has(sessionId)) {
      chatHistories.set(sessionId, new ChatMessageHistory());
    }
    const messageHistory = chatHistories.get(sessionId);

    // Embedding and LLM setup
    ollamaEmbedder = new OllamaEmbeddings({ model: "bge-m3:latest" });
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",
      temperature: 0,
      apiKey: process.env.GOOGLE_API_KEY,
    });

    // Vector search
    const queryVector = await ollamaEmbedder.embedQuery(queryText);
    const vectorSearchQuery = `
      CALL db.index.vector.queryNodes('text_chunks', 8, $vector)
      YIELD node, score
      RETURN node, score
    `;
    const vectorResults = await runQuery(vectorSearchQuery, { vector: queryVector });

    // Keyword-based search
    const keywords = queryText
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter(word => word.length > 3);

    let keywordSearchQuery = "";
    let keywordParams = {};

    if (keywords.length > 0) {
      keywordSearchQuery = `
        MATCH (chunk:TextChunk)
        WHERE ${keywords
          .map((_, i) => `toLower(chunk.chunk_text) CONTAINS $keyword${i}`)
          .join(" OR ")}
        RETURN chunk as node, 0.5 as score
        LIMIT 8
      `;
      keywords.forEach((keyword, i) => {
        keywordParams[`keyword${i}`] = keyword;
      });
    }

    const keywordResults = keywords.length > 0 ? await runQuery(keywordSearchQuery, keywordParams) : [];

    // Merge & deduplicate
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

    // Expand chunks
    const expandedNodes = new Set();
    const expandedResults = [];

    for (const result of combinedResults) {
      const nodeId = result.node.identity.toString();
      if (!expandedNodes.has(nodeId)) {
        expandedNodes.add(nodeId);
        expandedResults.push(result);

        const expandQuery = `
          MATCH (node)-[:NEXT_CHUNK*1..2]->(next_chunk)
          WHERE id(node) = $nodeId
          RETURN next_chunk as node, 0.4 as score
          LIMIT 3
        `;
        const expandedChunks = await runQuery(expandQuery, {
          nodeId: result.node.identity,
        });

        for (const expandedRecord of expandedChunks) {
          const expandedNode = expandedRecord.get("node");
          const expandedNodeId = expandedNode.identity.toString();
          if (!expandedNodes.has(expandedNodeId)) {
            expandedNodes.add(expandedNodeId);
            expandedResults.push({
              node: expandedNode,
              score: expandedRecord.get("score"),
            });
          }
        }
      }
    }

    const retrievedDocs = expandedResults
      .map(result => {
        const node = result.node;
        const chunkText = node.properties.chunk_text || "";
        const lawTitle = node.properties.law_title || "";
        const sectionNumber = node.properties.section_number || "";
        return `\n\n[${lawTitle} | Section ${sectionNumber}]\n ${chunkText}`;
      })
      .join("\n");

    // Print debug chunks
    // expandedResults.forEach(result => {
    //   const node = result.node;
    //   const chunkText = node.properties.chunk_text || "";
    //   const lawTitle = node.properties.law_title || "";
    //   const sectionNumber = node.properties.section_number || "";
    //   const score = result.score;
    //   console.log(
    //     `[Score: ${score.toFixed(3)}] [${lawTitle} | Section ${sectionNumber}] ${chunkText}`
    //   );
    //   console.log("-------------------------------------");
    // });

    // Chat history
    const pastMessages = await messageHistory.getMessages();
    const chatHistory = pastMessages
      .map(msg => (msg._getType() === "human" ? `Human: ${msg.content}` : `AI: ${msg.content}`))
      .join("\n");

    await messageHistory.addUserMessage(queryText);

    const prompt = `You are a law assistant for question-answering tasks on Bangladeshi legislature.
Use the following pieces of retrieved context to answer the question.
Don't start your answer with "Based on the retrieved context, ...". And if the question is a greeting or something irrelevant, answer accordingly.
If you don't know the answer, just say that you don't know. Don't say something like "The retrieved context is not enough to answer the question."

Chat history:
${chatHistory}

Question: "${queryText}"\n\nContext: "${retrievedDocs}\n\nAnswer:"
Answer:`;

    const response = await model.invoke(prompt);
    await messageHistory.addAIMessage(response.content);

  } catch (error) {
    console.error("Error handling /hybrid-chat request:", error);
    res.status(500).json({ error: "An error occurred while processing the request" });
  }
});


// Route: Get all laws sorted by publication date (latest first)
app.get("/laws", async (req, res) => {
  const { search = "", page = 1, limit = 50 } = req.query; // Get search, page, and limit from query parameters
  const offset = (page - 1) * limit; // Calculate the offset for pagination
  // Dynamically construct the query
  const query = `
 MATCH (l:Law)
 WHERE (l.title CONTAINS $search OR l.description CONTAINS $search)
 AND l.law_id <> 367
 RETURN l.law_id AS lawId, l.title AS title, l.has_chapters AS hasChapters, l.isRepealed AS isRepealed
 ORDER BY date(l.formatted_date) DESC
 SKIP $offset
 LIMIT $limit
`;
  try {
    // Execute the query with parameters
    const records = await runQuery(query, {
      search: search,
      offset: neo4j.int(offset),
      limit: neo4j.int(limit),
    });
    // Map the results but don't include repealed laws
    const laws = records
      .map((record) => ({
        lawId: record.get("lawId"),
        title: record.get("title"),
        hasChapters: record.get("hasChapters"),
        isRepealed: record.get("isRepealed"),
      }))
      .filter((law) => !law.isRepealed);
    res.json(laws);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch laws" });
  }
});
// Route: Get the field "hasChapters" and title for a specific law"
app.get("/laws/display/:lawId", async (req, res) => {
  const { lawId } = req.params;
  const query = `
 MATCH (l:Law {law_id: $lawId})
 RETURN l.title AS title, l.has_chapters AS hasChapters
 `;
  try {
    const records = await runQuery(query, { lawId: parseInt(lawId) });
    if (records.length === 0) {
      return res.status(404).json({ error: "Law not found" });
    }
    const lawDetails = {
      title: records[0].get("title"),
      hasChapters: records[0].get("hasChapters"),
    };
    res.json(lawDetails);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch law details" });
  }
});
// Route: Get specific fields (title, subtitle, publication_date, description) for a law
app.get("/laws/description/:lawId", async (req, res) => {
  const { lawId } = req.params;
  const query = `
 MATCH (l:Law {law_id: $lawId})
 RETURN l.title AS title, l.subtitle AS subtitle, l.publication_date AS publicationDate, l.description AS description
 `;
  try {
    const records = await runQuery(query, { lawId: parseInt(lawId) });
    if (records.length === 0) {
      return res.status(404).json({ error: "Law not found" });
    }
    const lawDetails = {
      title: records[0].get("title"),
      subtitle: records[0].get("subtitle"),
      publicationDate: records[0].get("publicationDate"),
      description: records[0].get("description"),
    };
    res.json(lawDetails);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch law details" });
  }
});
// Route: Get sections for a specific law
app.get("/laws/sections/:lawId", async (req, res) => {
  const { lawId } = req.params; // Get the lawId from route parameters
  // Query to fetch sections for the given law
  const query = `
 MATCH (l:Law {law_id: $lawId})-[:HAS_SECTION]->(s:Section)
 RETURN
 l.has_chapters AS hasChapters,
 l.has_parts AS hasParts,
 s.section_number AS sectionNumber,
 id(s) AS id,
 s.headline AS headline,
 s.section_key AS sectionKey,
 s.chapter_number AS chapterNumber,
 s.chapter_title AS chapterTitle,
 toInteger(apoc.text.replace(s.chapter_number, '\\D', '')) AS chapterOrder,
 toInteger(apoc.text.replace(s.section_number, '\\D.*', '')) AS sectionNumericOrder,
 apoc.text.replace(s.section_number, '\\d+', '') AS sectionAlphabeticOrder
 ORDER BY chapterOrder, id, sectionAlphabeticOrder
 `;
  try {
    // Execute the query with parameters
    const records = await runQuery(query, { lawId: parseInt(lawId) });
    // Map the results
    const sections = records.map((record) => ({
      section_number: record.get("sectionNumber"),
      section_key: record.get("sectionKey"),
      headline: record.get("headline"),
      chapter_number: record.get("chapterNumber"),
      chapter_title: record.get("chapterTitle"),
      // id: Integer { low: 157731, high: 0 } ; get the low value
      id: record.get("id").low,
    }));
    // console.log(sections);
    // Group sections by chapters if chapter_number exists
    const groupedSections = sections.reduce((acc, section) => {
      const chapterKey = section.chapter_number || "No Chapter";
      if (!acc[chapterKey]) {
        acc[chapterKey] = {
          chapter_title: section.chapter_title || "No Chapter Title",
          sections: [],
        };
      }
      acc[chapterKey].sections.push(section);
      return acc;
    }, {});
    res.json({
      lawId: parseInt(lawId),
      groupedSections,
    });
  } catch (error) {
    console.error("Error fetching sections:", error);
    res.status(500).json({ error: "Failed to fetch sections" });
  }
});
app.get("/constitution/sections", async (req, res) => {
  const lawId = 367; // The Constitution's law_id
  // Query to fetch sections for the Constitution
  const query = `
 MATCH (l:Law {law_id: $lawId})-[:HAS_SECTION]->(s:Section)
 RETURN
 l.has_chapters AS hasChapters,
 l.has_parts AS hasParts,
 s.section_number AS sectionNumber,
 id(s) AS id,
 s.headline AS headline,
 s.section_key AS sectionKey,
 s.chapter_number AS chapterNumber,
 s.chapter_title AS chapterTitle,
 s.part_number AS partNumber,
 s.part_title AS partTitle,
 toInteger(apoc.text.replace(s.part_number, '\\D', '')) AS partOrder,
 toInteger(apoc.text.replace(s.chapter_number, '\\D', '')) AS chapterOrder,
 toInteger(apoc.text.replace(s.section_number, '\\D.*', '')) AS sectionNumericOrder,
 apoc.text.replace(s.section_number, '\\d+', '') AS sectionAlphabeticOrder
 ORDER BY partOrder, chapterOrder, id, sectionAlphabeticOrder
 `;
  try {
    // Execute the query with parameters
    const records = await runQuery(query, { lawId });
    // Map the results
    const sections = records.map((record) => ({
      section_number: record.get("sectionNumber"),
      section_key: record.get("sectionKey"),
      headline: record.get("headline"),
      chapter_number: record.get("chapterNumber"),
      chapter_title: record.get("chapterTitle"),
      part_number: record.get("partNumber"),
      part_title: record.get("partTitle"),
      id: record.get("id").low, // Extracting low value from Neo4j integer object
    }));
    // console.log(sections);
    // Group sections by parts and chapters if they exist
    const groupedSections = sections.reduce((acc, section) => {
      const partKey = section.part_number || "No Part";
      if (!acc[partKey]) {
        acc[partKey] = {
          part_title: section.part_title || "No Part Title",
          chapters: {},
        };
      }
      const chapterKey = section.chapter_number || "No Chapter";
      if (!acc[partKey].chapters[chapterKey]) {
        acc[partKey].chapters[chapterKey] = {
          chapter_title: section.chapter_title || "No Chapter Title",
          sections: [],
        };
      }
      acc[partKey].chapters[chapterKey].sections.push(section);
      return acc;
    }, {});
    res.json({
      lawId,
      groupedSections,
    });
  } catch (error) {
    console.error("Error fetching Constitution sections:", error);
    res.status(500).json({ error: "Failed to fetch sections" });
  }
});
// Route: Get details for a specific section
app.get("/sections/:sectionKey", async (req, res) => {
  const { sectionKey } = req.params; // Get the sectionKey from route parameters
  // Query to fetch the section details
  const query = `
 MATCH (s:Section {section_key: $sectionKey})
 RETURN s.section_number AS sectionNumber,
 s.headline AS headline,
 s.markdown_text AS markdownText,
 s.parent_law_id AS parent_law_id
 `;
  try {
    // Execute the query with parameters
    const records = await runQuery(query, { sectionKey });
    if (records.length === 0) {
      return res.status(404).json({ error: "Section not found" });
    }
    // Extract the section details
    const section = records[0];
    const sectionDetails = {
      section_number: section.get("sectionNumber"),
      headline: section.get("headline"),
      markdown_text: section.get("markdownText"),
      parent_law_id: section.get("parent_law_id"),
    };
    res.json(sectionDetails);
  } catch (error) {
    console.error("Error fetching section details:", error);
    res.status(500).json({ error: "Failed to fetch section details" });
  }
});
// Route: Get all footnotes for a specific law by ID
app.get("/laws/:lawId/footnotes", async (req, res) => {
  const { lawId } = req.params;
  const query = `
 MATCH (l:Law {law_id: $lawId})<-[:FOOT_OF]-(f:Footnote)
 RETURN f.number AS number, f.text AS text
 ORDER BY toInteger(f.number)
 `;
  try {
    const records = await runQuery(query, { lawId: parseInt(lawId) });
    const footnotes = records.map((record) => ({
      number: record.get("number"),
      text: record.get("text"),
    }));
    res.json(footnotes);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch footnotes" });
  }
});
// Route to fetch footnotes for a given lawId and footnoteNumbers
app.post("/footnotes", async (req, res) => {
  const { lawId, footnoteNumbers } = req.body;
  // Validate request parameters
  if (
    !lawId ||
    !Array.isArray(footnoteNumbers) ||
    footnoteNumbers.length === 0
  ) {
    // console.log("HERe");
    return res.status(400).json({
      error: "Invalid parameters. 'lawId' and 'footnoteNumbers' are required.",
    });
  }
  try {
    // Neo4j query to fetch footnotes for the given lawId and footnoteNumbers
    const query = `
 MATCH (f:Footnote)-[:FOOT_OF]->(:Law {law_id: $lawId})
 WHERE f.number IN $footnoteNumbers
 RETURN f.number AS number, f.text AS text
 `;
    const params = {
      lawId: neo4j.int(lawId), // Ensure lawId is a number
      footnoteNumbers, // Array of footnote numbers
    };
    // Run the query
    const records = await runQuery(query, params);
    // console.log(records);
    // Map the results to a response format
    const footnotes = records.map((record) => ({
      number: record.get("number"),
      text: record.get("text"),
    }));
    // Send response
    res.status(200).json(footnotes);
  } catch (error) {
    console.error("Error fetching footnotes:", error);
    res
      .status(500)
      .json({ error: "Internal server error. Please try again later." });
  }
});

// Add an endpoint to clear chat history if needed
app.delete("/chat-history/:sessionId", (req, res) => {
  const { sessionId } = req.params;
  if (chatHistories.has(sessionId)) {
    chatHistories.delete(sessionId);
    res.status(200).json({ message: "Chat history cleared successfully" });
  } else {
    res.status(404).json({ error: "Session not found" });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
