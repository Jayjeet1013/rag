"use client";

import {
  Box,
  Button,
  Stack,
  TextField,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello! I'm the AI Rate My Professor Assistant. How may I assist you today?",
    },
  ]);
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const colors = {
    background: "#9900ff", // Darker background for a more modern look
    assistant: "#4caf50", // Green color for assistant messages
    user: "#2196f3", // Blue color for user messages
    text: "#ffffff", // White text color for contrast
    button: "#f50057", // Pink color for the send button
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    setIsLoading(true);

    const newMessage = { role: "user", content: message };
    setMessages((prevMessages) => [
      ...prevMessages,
      newMessage,
      { role: "assistant", content: "" },
    ]);

    setMessage("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      const assistantResponse =
        data.message ||
        "Apologies, I'm unable to process your request at this time.";

      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        { role: "assistant", content: assistantResponse },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: "assistant",
          content: "I'm sorry, an error occurred. Please try again later.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <Box
      width="100%"
      minHeight="100vh"
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
      className="animated-background" // Apply the animation class here
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        style={{ width: "100%", maxWidth: "600px" }}
      >
        <Box
          sx={{
            backgroundColor: "#004d40", // Solid dark green for header
            borderRadius: "12px",
            padding: "20px",
            marginBottom: "20px",
            textAlign: "center",
            boxShadow: 3,
          }}
        >
      
          <p className="font-semibold text-[20px] " style={{ color: colors.text }}>
            Welcome to AI Rate My Professor! 
          </p>
        </Box>
        <Stack
          direction="column"
          width="100%"
          maxWidth="600px"
          height={isMobile ? "80vh" : "600px"}
          border={`2px solid ${colors.button}`}
          borderRadius={theme.shape.borderRadius}
          p={2}
          spacing={3}
          sx={{
            backgroundColor: colors.background,
            overflow: "hidden",
            boxShadow: 3,
          }}
        >
          <Stack
            direction="column"
            spacing={2}
            flexGrow={1}
            overflow="auto"
            maxHeight="100%"
            sx={{
              overflowX: "hidden",
            }}
          >
            <AnimatePresence>
              {messages.map((msg, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                >
                  <Box
                    display="flex"
                    justifyContent={
                      msg.role === "assistant" ? "flex-start" : "flex-end"
                    }
                    sx={{ mb: 1 }}
                  >
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      transition={{
                        type: "spring",
                        stiffness: 400,
                        damping: 17,
                      }}
                    >
                      <Box
                        bgcolor={
                          msg.role === "assistant"
                            ? colors.assistant
                            : colors.user
                        }
                        color={colors.text}
                        borderRadius={5}
                        p={2}
                        sx={{
                          fontSize: isMobile ? "0.9rem" : "1rem",
                          wordBreak: "break-word",
                          whiteSpace: "pre-wrap",
                          overflowWrap: "break-word",
                        }}
                      >
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </Box>
                    </motion.div>
                  </Box>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </Stack>
          <Stack
            direction={isMobile ? "column" : "row"}
            spacing={2}
            alignItems="stretch"
          >
            <TextField
              label="Type your message..."
              fullWidth
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              variant="outlined"
              size={isMobile ? "small" : "medium"}
              InputProps={{
                style: {
                  fontSize: isMobile ? "0.9rem" : "1rem",
                  color: colors.text, // Text color for the input
                },
              }}
              InputLabelProps={{
                style: {
                  color: colors.text, // Text color for the label
                },
              }}
            />

            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              style={{ display: "flex" }}
            >
              <Button
                variant="contained"
                onClick={sendMessage}
                disabled={isLoading}
                fullWidth={isMobile}
                style={{
                  height: isMobile ? "40px" : "100%",
                  minWidth: isMobile ? "100%" : "100px",
                  backgroundColor: colors.button,
                  color: colors.text,
                  fontSize: isMobile ? "0.9rem" : "1rem",
                }}
              >
                {isLoading ? "Sending..." : "Send"}
              </Button>
            </motion.div>
          </Stack>
        </Stack>
      </motion.div>
    </Box>
  );
}
