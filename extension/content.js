// // File: extension/content.js

// console.log("AI Emotion Extension Loaded!");

// // Hàm gọi API Python
// async function getEmotion(text) {
//     try {
//         const response = await fetch("http://127.0.0.1:8000/predict", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ text: text })
//         });
//         const data = await response.json();
//         return data.emotions;
//     } catch (error) {
//         console.error("⚠️ Cannot connect to the server:", error);
//         return [];
//     }
// }

// // Hàm hiển thị nhãn lên web
// function displayLabel(element, emotions) {
//     if (emotions.length === 0) return;

//     // Lấy cảm xúc cao nhất
//     const topEmotion = emotions[0];
    
//     // Tạo thẻ hiển thị
//     const badge = document.createElement("div");
//     badge.innerText = `${topEmotion.label} ${Math.round(topEmotion.score * 100)}%`;
//     badge.className = "ai-emotion-badge";
    
//     // Tô màu badge tùy cảm xúc (Ví dụ đơn giản)
//     if (['joy', 'love', 'admiration'].includes(topEmotion.label)) badge.style.backgroundColor = "#4CAF50"; // Xanh
//     else if (['anger', 'disgust', 'hate'].includes(topEmotion.label)) badge.style.backgroundColor = "#F44336"; // Đỏ
//     else badge.style.backgroundColor = "#2196F3"; // Xanh dương

//     element.style.position = "relative";
//     element.appendChild(badge);
    
//     // Đánh dấu đã xử lý
//     element.setAttribute("data-ai-checked", "true");
// }

// // Hàm quét tin nhắn mới
// function scanMessages() {
//     // Selector này CẦN ĐIỀU CHỈNH tùy vào giao diện Zalo/FB tại thời điểm bạn chạy
//     // Đây là các class phổ biến thường chứa text tin nhắn
//     // FB/Messenger: div[dir="auto"], span, p
//     // Zalo: .card-text, .text-content
    
//     let selectors = "";
//     if (window.location.hostname.includes("zalo")) {
//         selectors = ".card-text, .text-content"; // Zalo Web
//     } else {
//         selectors = "div[dir='auto']"; // Messenger/FB
//     }

//     const messages = document.querySelectorAll(selectors);

//     messages.forEach(msg => {
//         // Chỉ xử lý tin nhắn chưa check và có nội dung dài > 2 ký tự
//         if (!msg.getAttribute("data-ai-checked") && msg.innerText.trim().length > 2) {
            
//             // Đánh dấu tạm để không gửi request trùng lặp liên tục
//             msg.setAttribute("data-ai-checked", "pending");

//             const text = msg.innerText;
            
//             // Gọi API
//             getEmotion(text).then(emotions => {
//                 if (emotions.length > 0) {
//                     displayLabel(msg, emotions);
//                 } else {
//                     msg.setAttribute("data-ai-checked", "true"); // Đánh dấu xong dù không có emotion
//                 }
//             });
//         }
//     });
// }

// // Chạy scan mỗi 2 giây (Cách đơn giản nhất thay vì MutationObserver phức tạp)
// setInterval(scanMessages, 2000);




// File: extension/content.js

// console.log("AI Emotion Extension Loaded!");

// const EMOTION_EMOJIS = {
//     "admiration": "🤩",    "amusement": "😂",    "anger": "😡",
//     "annoyance": "😒",     "approval": "👍",     "caring": "🤗",
//     "confusion": "😕",     "curiosity": "🤔",    "desire": "😍",
//     "disappointment": "😞", "disapproval": "👎",  "disgust": "🤢",
//     "embarrassment": "😳",  "excitement": "😆",   "fear": "😰",
//     "gratitude": "🙏",     "grief": "😭",        "joy": "😊",
//     "love": "❤️",          "nervousness": "😬",  "optimism": "🤞",
//     "pride": "😎",         "realization": "💡",   "relief": "😌",
//     "remorse": "😔",       "sadness": "😢",      "surprise": "😲",
//     "neutral": "🐣"
// };

// const SHORT_TEXT_THRESHOLD = 15; 

// async function getEmotion(text) {
//     try {
//         const response = await fetch("http://127.0.0.1:8000/predict", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ text: text })
//         });
//         const data = await response.json();
//         return data.emotions;
//     } catch (error) {
//         return [];
//     }
// }

// function displayLabel(element, emotions, originalText) {
//     if (emotions.length === 0) return;

//     element.style.overflow = "visible"; 
//     if (getComputedStyle(element).position === 'static') {
//         element.style.position = "relative";
//     }

//     const topEmotion = emotions[0];
//     const label = topEmotion.label;
//     const score = Math.round(topEmotion.score * 100);
    
//     // Xóa nhãn cũ
//     const oldBadge = element.querySelector('.ai-emotion-badge');
//     if (oldBadge) oldBadge.remove();

//     const badge = document.createElement("div");
//     badge.className = "ai-emotion-badge";

//     const cleanLength = originalText.replace(/\s+/g, ' ').trim().length;

//     if (cleanLength < SHORT_TEXT_THRESHOLD) {
//         // === STYLE EMOJI KHÔNG NỀN, SIZE TO ===
//         badge.innerText = EMOTION_EMOJIS[label] || "🤖";
        
//         Object.assign(badge.style, {
//             // 1. Size emoji
//             fontSize: "18px", 
            
//             // 2. Remove background and border
//             backgroundColor: "transparent",
//             boxShadow: "none",
//             border: "none",
//             width: "auto",
//             height: "auto",
            
//             // 3. Tạo bóng đổ cho chính Emoji để nổi bật trên nền chat
//             textShadow: "0 2px 5px rgba(0,0,0,0.3)",
            
//             // 4. Vị trí "Nửa trong nửa ngoài"
//             // Với size 24px, đặt -12px là tâm nằm ngay mép góc
//             position: "absolute",
//             top: "-12px",     
//             left: "-12px",    
//             zIndex: "999999",
//             cursor: "default"
//         });

//     } else {
//         // === STYLE TEXT ===
//         badge.innerText = `${label} ${score}%`;
        
//         Object.assign(badge.style, {
//             fontSize: "10px", 
//             color: "white",
//             padding: "1px 6px",
//             borderRadius: "8px",
//             position: "absolute",
//             bottom: "-8px",
//             right: "0",
//             zIndex: "99999",
//             whiteSpace: "nowrap",
//             boxShadow: "0 1px 2px rgba(0,0,0,0.2)"
//         });
        
//         if (['joy', 'love', 'admiration', 'excitement'].includes(label)) badge.style.backgroundColor = "#4CAF50"; 
//         else if (['anger', 'disgust', 'hate'].includes(label)) badge.style.backgroundColor = "#F44336"; 
//         else badge.style.backgroundColor = "#2196F3"; 
//     }

//     element.appendChild(badge);
//     element.setAttribute("data-ai-checked", "true");
// }

// function scanMessages() {
//     let selectors = "";
//     if (window.location.hostname.includes("zalo")) {
//         selectors = ".card-text, .text-content"; 
//     } else {
//         selectors = "div[dir='auto']"; 
//     }
//     const messages = document.querySelectorAll(selectors);
//     messages.forEach(msg => {
//         const cleanText = msg.innerText;
//         if (!msg.getAttribute("data-ai-checked") && cleanText.trim().length > 0) {
//             msg.setAttribute("data-ai-checked", "pending");
//             getEmotion(cleanText).then(emotions => {
//                 if (emotions.length > 0) {
//                     displayLabel(msg, emotions, cleanText); 
//                 } else {
//                     msg.setAttribute("data-ai-checked", "true");
//                 }
//             });
//         }
//     });
// }

// setInterval(scanMessages, 2000);

// File: extension/content.js

console.log("AI Emotion Extension Loaded (Color Coded)!");

const EMOTION_EMOJIS = {
    "admiration": "🤩",    "amusement": "😂",    "anger": "😡",
    "annoyance": "😒",     "approval": "👍",     "caring": "🤗",
    "confusion": "😕",     "curiosity": "🤔",    "desire": "😍",
    "disappointment": "😞", "disapproval": "👎",  "disgust": "🤢",
    "embarrassment": "😳",  "excitement": "🥳",   "fear": "😰",
    "gratitude": "🙏",     "grief": "😭",        "joy": "😊",
    "love": "❤️",          "nervousness": "😬",  "optimism": "🤞",
    "pride": "😎",         "realization": "💡",   "relief": "😌",
    "remorse": "😔",       "sadness": "😢",      "surprise": "😲",
    "neutral": "🐣"
};

// --- PHÂN LOẠI CẢM XÚC ĐỂ TÔ MÀU ---
const NEGATIVE_EMOTIONS = [
    'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
];

const POSITIVE_EMOTIONS = [
    'admiration', 'amusement', 'approval', 'caring', 'desire', 
    'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'
];

// Các từ chào hỏi
const GREETING_WORDS = ["hello", "hi", "hey", "hola", "xin chào", "chào", "chào bạn", "alo"];

const SHORT_TEXT_THRESHOLD = 15; 

async function getEmotion(text) {
    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        return data.emotions;
    } catch (error) {
        return [];
    }
}

function displayLabel(element, emotions, originalText) {
    if (emotions.length === 0) return;

    element.style.overflow = "visible"; 
    if (getComputedStyle(element).position === 'static') {
        element.style.position = "relative";
    }

    const topEmotion = emotions[0];
    const label = topEmotion.label;
    const score = Math.round(topEmotion.score * 100);
    
    // Xóa nhãn cũ
    const oldBadge = element.querySelector('.ai-emotion-badge');
    if (oldBadge) oldBadge.remove();

    const badge = document.createElement("div");
    badge.className = "ai-emotion-badge";

    const cleanLength = originalText.replace(/\s+/g, ' ').trim().length;
    const lowerText = originalText.toLowerCase().replace(/[!.,?]/g, "").trim();

    // Logic xác định màu sắc chủ đạo
    let colorCode = "#2196F3"; // Mặc định: Xanh dương (Neutral, Surprise...)
    let glowColor = "rgba(0,0,0,0.3)"; // Mặc định bóng đen nhẹ

    if (NEGATIVE_EMOTIONS.includes(label)) {
        colorCode = "#D32F2F"; // ĐỎ ĐẬM (Negative)
        glowColor = "rgba(211, 47, 47, 0.6)"; // Bóng đỏ
    } else if (POSITIVE_EMOTIONS.includes(label)) {
        colorCode = "#388E3C"; // XANH LÁ ĐẬM (Positive)
        glowColor = "rgba(56, 142, 60, 0.6)"; // Bóng xanh
    }

    // Xử lý chào hỏi
    let displayEmoji = EMOTION_EMOJIS[label] || "🤖";
    if (GREETING_WORDS.includes(lowerText)) {
        displayEmoji = "👋";
        colorCode = "#FF9800"; // Chào hỏi cho màu Cam
        glowColor = "rgba(255, 152, 0, 0.6)";
    }

    if (cleanLength < SHORT_TEXT_THRESHOLD) {
        // === HIỆN EMOJI ===
        badge.innerText = displayEmoji;
        
        Object.assign(badge.style, {
            fontSize: "18px", 
            backgroundColor: "transparent",
            boxShadow: "none",
            border: "none",
            
            // TẠO VIỀN SÁNG THEO MÀU CẢM XÚC
            textShadow: `0 0 5px ${glowColor}, 0 0 10px ${glowColor}`,
            
            position: "absolute",
            top: "-12px",     
            left: "-12px",    
            zIndex: "999999",
            cursor: "default"
        });

    } else {
        // === HIỆN TEXT BADGE (TEXT DÀI) ===
        badge.innerText = `${label} ${score}%`;
        
        Object.assign(badge.style, {
            fontSize: "10px", 
            color: "white",
            padding: "2px 8px",
            borderRadius: "8px",
            position: "absolute",
            bottom: "-8px",
            right: "0",
            zIndex: "99999",
            whiteSpace: "nowrap",
            boxShadow: "0 1px 2px rgba(0,0,0,0.2)",
            
            // GÁN MÀU NỀN THEO PHÂN LOẠI
            backgroundColor: colorCode 
        });
    }

    element.appendChild(badge);
    element.setAttribute("data-ai-checked", "true");
}

function scanMessages() {
    let selectors = "";
    if (window.location.hostname.includes("zalo")) {
        selectors = ".card-text, .text-content"; 
    } else {
        selectors = "div[dir='auto']"; 
    }
    const messages = document.querySelectorAll(selectors);
    messages.forEach(msg => {
        const cleanText = msg.innerText;
        if (!msg.getAttribute("data-ai-checked") && cleanText.trim().length > 0) {
            msg.setAttribute("data-ai-checked", "pending");
            getEmotion(cleanText).then(emotions => {
                if (emotions.length > 0) {
                    displayLabel(msg, emotions, cleanText); 
                } else {
                    msg.setAttribute("data-ai-checked", "true");
                }
            });
        }
    });
}

setInterval(scanMessages, 2000);