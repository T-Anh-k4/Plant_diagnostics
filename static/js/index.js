let predictionResult = "";
let currentConversation = [];

// Khởi tạo sự kiện
document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("new-chat-btn").addEventListener("click", createNewChat);
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") sendMessage();
});

async function sendMessage() {
    const input = document.getElementById("user-input");
    const userMessage = input.value.trim();
    if (userMessage === "") return;

    const chatBox = document.getElementById("chat-box");

    // Hiển thị tin nhắn người dùng
    displayMessage(userMessage, "user-message");
    currentConversation.push({ sender: "user", text: userMessage });
    input.value = "";

    // Hiển thị trạng thái "đang gõ"
    const typingIndicator = displayMessage("Chatbot đang trả lời...", "bot-message typing");

    try {
        if (userMessage.toLowerCase().includes("dự đoán") || userMessage.toLowerCase().includes("kết quả")) {
            // Xử lý yêu cầu kết quả dự đoán
            removeTypingIndicator(typingIndicator);
            const responseText = predictionResult
                ? `Kết quả dự đoán: ${predictionResult}`
                : "Chưa có kết quả dự đoán nào. Vui lòng tải lên ảnh trước.";
            displayMessage(responseText, "bot-message");
            currentConversation.push({ sender: "bot", text: responseText });
        } else {
            // Gửi yêu cầu đến server
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage })
            });

            if (!response.ok) {
                throw new Error(`Lỗi server: ${response.status}`);
            }

            const data = await response.json();
            removeTypingIndicator(typingIndicator);

            const botResponse = data.response || "Xin lỗi, tôi không hiểu yêu cầu của bạn.";
            displayMessage(botResponse, "bot-message");
            currentConversation.push({ sender: "bot", text: botResponse });
        }
    } catch (error) {
        removeTypingIndicator(typingIndicator);
        const errorMessage = `Lỗi: ${error.message}`;
        displayMessage(errorMessage, "bot-message error");
        currentConversation.push({ sender: "bot", text: errorMessage });
        console.error("Lỗi khi gửi tin nhắn:", error);
    }
}

// Hiển thị tin nhắn
function displayMessage(text, className) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${className}`;
    messageDiv.innerText = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// Xóa indicator "đang gõ"
function removeTypingIndicator(indicator) {
    if (indicator && indicator.parentNode) {
        indicator.parentNode.removeChild(indicator);
    }
}

// Tạo cuộc trò chuyện mới
function createNewChat() {
    currentConversation = [];
    document.getElementById("chat-box").innerHTML = "";
    document.getElementById("user-input").value = "";
}