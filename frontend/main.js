// Funktion zum Formatieren der Bevölkerungszahlen
function formatPopulation(population) {
    return population.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

function predictPopulation() {
    // Wert aus dem Eingabefeld abrufen
    var year = document.getElementById('yearInput').value;

    // AJAX-Anfrage an den Flask-Server senden, um Vorhersagen zu erhalten
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/predict?year=' + year, true);

    // Callback-Funktion für den Abschluss der Anfrage
    xhr.onload = function () {
        if (xhr.status === 200) {
            // JSON-Daten von der Antwort erhalten
            var data = JSON.parse(xhr.responseText);

            // Vorhersagen auf der Seite anzeigen
            document.getElementById('predictionResult').innerHTML = `
            <p>Vorhersage mit dem linearen Modell: ${formatPopulation(data.prediction_LinearModel)}</p>
            <p>Vorhersage mit dem Random-Forest-Modell: ${formatPopulation(data.prediction_RandomForest)}</p>
            `;
        } else {
            console.error('Fehler beim Abrufen der Vorhersagen.');
        }
    };

    // Fehlerbehandlung für die AJAX-Anfrage
    xhr.onerror = function () {
        console.error('Fehler beim Senden der Anfrage.');
    };

    // Anfrage senden
    xhr.send();
}

// Funktion zum Abrufen und Anzeigen der MSE-Werte
function getMSE() {
    // AJAX-Anfrage an den Flask-Server senden, um MSE-Werte zu erhalten
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/mse', true);

    // Callback-Funktion für den Abschluss der Anfrage
    xhr.onload = function () {
        if (xhr.status === 200) {
            // JSON-Daten von der Antwort erhalten
            var mseData = JSON.parse(xhr.responseText);

            // MSE-Werte auf der Seite anzeigen
            var mseHtml = '';
            for (var model in mseData) {
                mseHtml += `<h3>${model}</h3>`;
                for (var fileType in mseData[model]) {
                    mseHtml += `<p>${fileType}: ${mseData[model][fileType]}</p>`;
                }
            }
            document.getElementById('mseResults').innerHTML = mseHtml;
        } else {
            console.error('Fehler beim Abrufen der MSE-Werte.');
        }
    };

    // Fehlerbehandlung für die AJAX-Anfrage
    xhr.onerror = function () {
        console.error('Fehler beim Senden der Anfrage.');
    };

    // Anfrage senden
    xhr.send();
}

// Funktion zum Anzeigen oder Ausblenden der MSE-Werte
function toggleMSE() {
    var mseResults = document.getElementById('mseResults');
    if (mseResults.style.display === 'none') {
        getMSE();
        mseResults.style.display = 'block';
    } else {
        mseResults.style.display = 'none';
    }
}

// Aufrufen der Funktion zum Abrufen der MSE-Werte beim Laden der Seite
window.onload = function () {
    getMSE();
};
