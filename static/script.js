const language = document.getElementById("language");
const newsText = document.getElementById("newsText");
const detectButton = document.getElementById("detectButton");
const prediction = document.getElementById("prediction");

function showResult(data) {
    prediction.textContent = `${data.prediction} (${data.confidence}% model confidence)`;
}

detectButton.addEventListener("click", async function () {
    const selectedLanguage = language.value;
    const text = newsText.value.trim();

    if (selectedLanguage === "") {
        prediction.textContent = "Please select a language.";
        return;
    }

    if (text === "") {
        prediction.textContent = "Please enter news text.";
        return;
    }

    detectButton.disabled = true;
    detectButton.textContent = "Checking...";
    prediction.textContent = "Checking language and news...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ language: selectedLanguage, text })
        });

        const data = await response.json();

        if (!response.ok) {
            prediction.textContent = data.error || "An error occurred.";
            return;
        }

        showResult(data);
    } catch (error) {
        prediction.textContent = "Unable to connect to the server.";
    } finally {
        detectButton.disabled = false;
        detectButton.textContent = "Detect News";
    }
});
