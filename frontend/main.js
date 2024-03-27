function predictPopulation() {
    // Wert aus dem Eingabefeld für das Jahr erhalten
    var year = document.getElementById("yearInput").value;

    // GET-Anfrage an die Flask-Route senden, um die Vorhersagen zu erhalten
    fetch("/predict?year=" + year)
        .then(response => response.json())
        .then(data => {
            // Vorhersagen anzeigen
            document.getElementById("predictionResult").innerHTML = `
                <p>Trained model: ${data.prediction_trainingdata.toLocaleString()} Einwohner</p>
                <p>Validated model: ${data.prediction_validationdata.toLocaleString()} Einwohner</p>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
