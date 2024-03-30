function predictPopulation() {
    // Wert aus dem Eingabefeld für das Jahr erhalten
    var year = document.getElementById("yearInput").value;

    // GET-Anfrage an die Flask-Route senden, um die Vorhersagen und MSE-Werte zu erhalten
    fetch("/predict?year=" + year)
        .then(response => response.json())
        .then(data => {
            // Vorhersagen anzeigen
            document.getElementById("predictionResult").innerHTML = `
                <p>Vorhersage Linear Model: ${data.prediction_Linearmodel.toLocaleString()} Einwohner</p>
                <p>Vorhersage Random Forest: ${data.prediction_RandomForest.toLocaleString()} Einwohner</p>
            `;
            
            // GET-Anfrage an die Flask-Route senden, um die MSE-Werte zu erhalten
            fetch("/mse")
                .then(response => response.json())
                .then(mseData => {
                    // MSE-Werte anzeigen
                    document.getElementById("mseResult").innerHTML = `
                        <p>Trainings-MSE Linear Model: ${mseData.train_mse_lm}</p>
                        <p>Validierungs-MSE Linear Model: ${mseData.valid_mse_lm}</p>
                        <p>Trainings-MSE Random Forest: ${mseData.train_mse_rf}</p>
                        <p>Validierungs-MSE Random Forest: ${mseData.valid_mse_rf}</p>
                    `;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
