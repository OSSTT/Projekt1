function predictPopulation() {
    // Wert aus dem Eingabefeld für das Jahr erhalten
    var year = document.getElementById("yearInput").value;

    // GET-Anfrage an die Flask-Route senden, um die Vorhersagen zu erhalten
    fetch("/predict?year=" + year)
        .then(response => response.json())
        .then(data => {
            // Vorhersagen anzeigen
            document.getElementById("predictionResult").innerHTML = `
                <p>Vorhersage Training-Model: ${data.prediction_trainingdata}</p>
                <p>Vorhersage Validation-Model: ${data.prediction_validationdata}</p>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
