import { NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = (userMessage) => `
You are a Rate My Professor agent designed to assist students in finding the best classes by answering their questions. For each user query, analyze the question to understand the specific needs, such as preferred teaching styles, course difficulty, or professor ratings. Return the top 5 professors who best match the criteria based on student reviews, course content, and teaching effectiveness. Use the information about these professors to provide a detailed and helpful response to the user’s question. Always prioritize accurate and relevant information to ensure students make informed decisions about their class selections.

[
    {"professor": "Dr. Alice Johnson", "subject": "Mathematics", "stars": 5, "review": "Dr. Johnson is an exceptional teacher who makes complex topics easy to understand. Highly recommended!"},
    {"professor": "Prof. Robert Smith", "subject": "Physics", "stars": 4, "review": "Prof. Smith is very knowledgeable and passionate about the subject. Classes can be challenging but rewarding."},
    {"professor": "Dr. Emily Davis", "subject": "Biology", "stars": 3, "review": "Dr. Davis is good, but her lectures sometimes lack engagement. The content is thorough, though."},
    {"professor": "Prof. Michael Lee", "subject": "Computer Science", "stars": 5, "review": "Prof. Lee is a fantastic professor who always provides real-world examples and hands-on learning opportunities."},
    {"professor": "Dr. Sarah Wilson", "subject": "Chemistry", "stars": 4, "review": "Dr. Wilson is very engaging and makes chemistry interesting. Some of the labs can be tough, but she's always there to help."},
    {"professor": "Prof. Daniel Brown", "subject": "History", "stars": 3, "review": "Prof. Brown is knowledgeable but his lectures can be a bit dry. The material is important, though."},
    {"professor": "Dr. Laura Martinez", "subject": "Economics", "stars": 5, "review": "Dr. Martinez is an excellent professor who explains economic concepts clearly and is very approachable."},
    {"professor": "Prof. James Taylor", "subject": "Literature", "stars": 4, "review": "Prof. Taylor has a deep understanding of literature and provides insightful analyses. The class discussions are very enriching."},
    {"professor": "Dr. Elizabeth Anderson", "subject": "Philosophy", "stars": 5, "review": "Dr. Anderson is an inspiring teacher with a knack for making philosophical concepts accessible and engaging."},
    {"professor": "Prof. William Harris", "subject": "Political Science", "stars": 3, "review": "Prof. Harris is knowledgeable, but the lectures can sometimes be unorganized. The subject matter is interesting, though."},
    {"professor": "Dr. Jessica Clark", "subject": "Engineering", "stars": 4, "review": "Dr. Clark is a great professor with practical insights into engineering. Her classes are challenging but rewarding."},
    {"professor": "Prof. David Robinson", "subject": "Art History", "stars": 5, "review": "Prof. Robinson is passionate about art and provides a deep understanding of historical contexts. Highly engaging classes."},
    {"professor": "Dr. Nancy Lewis", "subject": "Sociology", "stars": 3, "review": "Dr. Lewis is good at explaining concepts, but the pace of the class can be slow at times."},
    {"professor": "Prof. Charles Walker", "subject": "Statistics", "stars": 4, "review": "Prof. Walker makes statistics approachable and provides helpful examples. The assignments can be demanding, though."},
    {"professor": "Dr. Karen Hall", "subject": "Geology", "stars": 5, "review": "Dr. Hall is very enthusiastic about geology and her field trips are fantastic. The class is both informative and fun."},
    {"professor": "Prof. Steven Young", "subject": "Linguistics", "stars": 4, "review": "Prof. Young is knowledgeable and passionate about linguistics. The class discussions are lively and engaging."},
    {"professor": "Dr. Olivia Wright", "subject": "Psychology", "stars": 3, "review": "Dr. Wright is a good professor, but the lectures can sometimes feel repetitive. The material is still valuable."},
    {"professor": "Prof. Richard King", "subject": "Pharmacy", "stars": 4, "review": "Prof. King is experienced and provides practical knowledge in pharmacy. His lectures are thorough and well-structured."},
    {"professor": "Dr. Sophia Scott", "subject": "Music", "stars": 5, "review": "Dr. Scott is an exceptional music professor who fosters creativity and offers valuable feedback. The classes are inspiring."},
    {"professor": "Prof. John Martinez", "subject": "Environmental Science", "stars": 4, "review": "Prof. Martinez is very passionate about environmental issues and provides practical insights. The coursework is engaging."},
    {"professor": "Dr. Emily Adams", "subject": "Mathematics", "stars": 3, "review": "Dr. Adams is knowledgeable, but her teaching style may not suit everyone. The material is comprehensive, though."},
    {"professor": "Prof. Anthony Nelson", "subject": "Engineering", "stars": 5, "review": "Prof. Nelson is an outstanding professor who provides real-world applications of engineering principles. Highly recommended."}
]


User: ${userMessage}
Assistant:
`;

const apiKey = process.env.GOOGLE_API_KEY;

if (!apiKey) {
  console.error("API key is missing");
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

export async function POST(req) {
  try {
    const data = await req.json();
    const { message: userMessage } = data;

    if (!userMessage) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    const prompt = systemPrompt(userMessage);

    const result = await model.generateContent(prompt);
    const response = await result.response.text();

    return NextResponse.json({ message: response });
  } catch (error) {
    console.error("Error generating response:", error);
    return NextResponse.json(
      { error: "Error generating response" },
      { status: 500 }
    );
  }
}
