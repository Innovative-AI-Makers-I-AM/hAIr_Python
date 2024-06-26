import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([{ text: "챗봇이 시작합니다...", user: "bot" }]);
  const [input, setInput] = useState("");

  useEffect(() => {
    axios.get('http://localhost:8000/init_chat')
      .then(response => {
        setMessages([...messages, { text: response.data.response, user: "bot" }]);
      });
  }, []);

  const sendMessage = () => {
    if (input.trim() === "") return;
    
    setMessages([...messages, { text: input, user: "user" }]);
    axios.post('http://localhost:8000/chat', { message: input })
      .then(response => {
        setMessages(prevMessages => [...prevMessages, { text: response.data.response, user: "bot" }]);
      });

    setInput("");
  };

  const endChat = () => {
    axios.post('http://localhost:8000/end_chat')
      .then(response => {
        setMessages([...messages, { text: response.data.response, user: "bot" }]);
      });
  };

  return (
    <div className="App">
      <div id="chatbox">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.user}-message`}>{msg.text}</div>
        ))}
      </div>
      <input 
        type="text" 
        id="userInput" 
        placeholder="메시지를 입력하세요" 
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button onClick={sendMessage}>보내기</button>
      <button onClick={endChat}>대화 종료</button>
    </div>
  );
}

export default App;