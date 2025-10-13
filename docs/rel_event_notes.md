# RelBench "rel-event" Übersicht und Pipeline-Ausgabe

## Datenbankbeschreibung
- Event Recommendation Datenbank aus der Hangtime-App mit vollständig anonymisierten Nutzer-, Ereignis- und Beziehungsdaten.
- 5 Tabellen, 41.328.337 Zeilen, 128 Spalten.
- Zeitliche Abdeckung von 1. Januar 1912 bis zu Validierungs- und Testzeitpunkten am 21. bzw. 29. November 2012.
- Evaluationsaufgaben umfassen u. a. die Regressionsaufgabe `user-attendance` (Vorhersage der Zahl bestätigter Eventteilnahmen in den nächsten sieben Tagen, Metrik: MAE).

## Einordnung der Pipeline-Ausgabe
Die Featuretools + TabPFN-RelBench-Pipeline bereitet die `train`-, `val`- und `test`-Splits auf, meldet deren Beobachtungsanzahl und ob Zielwerte verfügbar sind. Nur Splits mit "labels available" liefern Metriken; Splits ohne Zielwerte melden `No targets provided …` und überspringen die Evaluation. Für Regressionen werden pro Split R², MAE und RMSE ausgegeben.

## Warum der Test-Split keine Kennzahlen zeigt
- RelBench blendet beim `test`-Split vieler Aufgaben (u. a. `rel-event`) die Zielwerte aus, damit die offizielle Benchmark-Plattform die Auswertung übernimmt. Deshalb meldet die Pipeline `labels hidden` und später `[test] No targets provided; skipping evaluation.`
- Das bedeutet: Für diesen Split findet lokal **keine** Metrikberechnung statt, weil keine Ground-Truth-Werte vorhanden sind. Training und Validierung laufen trotzdem normal durch.

## Wie erhalte ich dennoch Test-Ergebnisse?
1. **Vorhersagen speichern:** Starte die Pipeline mit `--save-predictions <ordner>`. Für jeden Split entsteht eine CSV-Datei (z. B. `test_predictions.csv`) mit `observation_id`, der jeweiligen Entitäts-ID sowie Modellprognosen. Bei Klassifikationsaufgaben werden zusätzlich Klassenwahrscheinlichkeiten pro Label abgelegt.
2. **Datei für RelBench aufbereiten:** Prüfe im offiziellen RelBench-Leitfaden, welche Spalten und welches Format für die jeweilige Aufgabe gefordert sind. Die erzeugte CSV lässt sich dort bei Bedarf anpassen (z. B. Spalten umbenennen).
3. **Auf RelBench einreichen:** Lade die Vorhersagedatei im RelBench-Portal hoch. Erst dort erfolgt die Auswertung gegen die versteckten Test-Labels, und du erhältst die offiziellen Kennzahlen (z. B. MAE für `user-attendance`).

Tipp: Nutze während der Modellentwicklung den Validierungssplit (`val`), da dessen Zielwerte sichtbar bleiben und du sofort siehst, wie sich Änderungen an Featuretools- oder TabPFN-Parametern auf die Metriken auswirken. Sobald das Modell steht, exportierst du die `test`-Vorhersagen und reichst sie beim Benchmark ein.
