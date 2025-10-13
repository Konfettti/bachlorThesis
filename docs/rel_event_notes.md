# RelBench "rel-event" Übersicht und Pipeline-Ausgabe

## Datenbankbeschreibung
- Event Recommendation Datenbank aus der Hangtime-App mit vollständig anonymisierten Nutzer-, Ereignis- und Beziehungsdaten.
- 5 Tabellen, 41.328.337 Zeilen, 128 Spalten.
- Zeitliche Abdeckung von 1. Januar 1912 bis zu Validierungs- und Testzeitpunkten am 21. bzw. 29. November 2012.
- Evaluationsaufgaben umfassen u. a. die Regressionsaufgabe `user-attendance` (Vorhersage der Zahl bestätigter Eventteilnahmen in den nächsten sieben Tagen, Metrik: MAE).

## Einordnung der Pipeline-Ausgabe
Die Featuretools + TabPFN-RelBench-Pipeline bereitet die `train`-, `val`- und `test`-Splits auf, meldet deren Beobachtungsanzahl und ob Zielwerte verfügbar sind. Nur Splits mit "labels available" liefern Metriken; Splits ohne Zielwerte melden `No targets provided …` und überspringen die Evaluation. Für Regressionen werden pro Split R², MAE und RMSE ausgegeben.
