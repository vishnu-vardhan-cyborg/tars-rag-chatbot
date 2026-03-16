async function sendMessage(){

let input = document.getElementById("userInput")
let message = input.value

let chatbox = document.getElementById("chatbox")

chatbox.innerHTML += "<p class='user'>USER: "+message+"</p>"

let API_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000/chat"
  : "/chat"

let response = await fetch(API_URL,{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({question:message})
})

let data = await response.json()

let formatted = data.answer.replace(/\n/g,"<br>")

chatbox.innerHTML += "<p class='bot'>TARS:<br>"+formatted+"</p>"

input.value=""

chatbox.scrollTop = chatbox.scrollHeight

/* UPDATE METRICS */

document.getElementById("faithfulness").innerText = data.metrics.faithfulness
document.getElementById("relevance").innerText = data.metrics.answer_relevance
document.getElementById("precision").innerText = data.metrics.context_precision
document.getElementById("recall").innerText = data.metrics.context_recall

document.getElementById("faithfulness-bar").style.width = (data.metrics.faithfulness * 100) + "%"
document.getElementById("relevance-bar").style.width = (data.metrics.answer_relevance * 100) + "%"
document.getElementById("precision-bar").style.width = (data.metrics.context_precision * 100) + "%"
document.getElementById("recall-bar").style.width = (data.metrics.context_recall * 100) + "%"

}
