# Decision-Tree

**Anleitung zum Ausführen des Notebooks:**
1. Klicken Sie auf das MyBinder Batch ↓
2. Warten Sie bis sich das Notebook geöffnet hat (kann bis zu 15 Minuten dauern)
3. Das Notebook wird direkt aufgerufen und kann durch Verwenden des Play-Button schrittweise ausgeführt werden
   
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MichaelFranLu/Decision-Tree/master?labpath=3-Decision_Trees_und_Random_Forests_Projekt-Loesung.ipynb)

**Update:**
Ausführen in Google Colab ebenfalls möglich & performanter:
1. Klicken Sie auf den Google Colab Batch ↓
2. Warten Sie bis sich das Notebook geöffnet hat (maximal 10 Sekunden)
3. Das Notebook wird direkt aufgerufen und kann durch Verwenden des Play-Button schrittweise ausgeführt werden

<a target="_blank" href="https://colab.research.google.com/github/MichaelFranLu/Decision-Tree/blob/main/3-Decision_Trees_und_Random_Forests_Projekt-Loesung.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


**Dokumentation - Logistische Regression**

Dieser Code erstellt prädiktive Modelle, basierend auf Entscheidungsbäumen und Random Forests, um die Wahrscheinlichkeit der vollständigen Rückzahlung von Darlehen durch Kreditnehmer auf LendingClub.com zu bestimmen. Er nutzt öffentliche Daten von 2007 bis 2010 und zielt darauf ab, Kreditgebern bei der Auswahl zuverlässiger Kreditnehmer zu helfen.

1. credit.policy: Binärer Indikator (1 oder 0), der angibt, ob der Kunde die Risikobewertung bestanden hat oder nicht
2. purpose: Bezeichnet den Zweck des Kredits. Die möglichen Werte sind “credit_card”, “debt_consolidation”, “educational”, “major_purchase”, “small_business” und “all_other”
3. int.rate: Ist der Zinssatz des Kredits als Anteil. Ein höherer Zinssatz wird an Kreditnehmer vergeben, die von LendingClub.com als riskanter eingestuft werden
4. installment: Die monatliche Zahlung, die der Kreditnehmer leistet, wenn der Kredit finanziert wird
5. log.annual.inc: Logarithmus des vom Kreditnehmer angegebenen jährlichen Einkommens
6. dti: Ist die “debt-to-income” Rate des Kreditnehmers, berechnet als Kredit geteilt durch jährliches Einkommen
7. fico: FICO-Kreditscore des Kreditnehmers
8. days.with.cr.line: Anzahl der Tage, an denen der Kunde einen Dispokredit hatte
9. revol.bal: Bilanz am Ende eines Kreditkartenabrechnungszeitraums
10. revol.util: Erstatteter Anteil am Gesamtkredit.
11. inq.last.6mths: Anzahl der Anfragen, die Kreditgeber in den letzten 6 Monaten an den Kreditnehmer gestellt haben
12. delinq.2yrs: Anzahl der Vorkommnisse eines Verzugs von über 30 Tagen innerhalb der letzten 2 Jahre
13. pub.rec: Dies ist die Anzahl der negativen Einträge (wie Bankrott, Steuerverzug, Verurteilungen usw.) des Kreditnehmers.

**Benötigte Libraries instalieren:**
- pandas
- numpy
- matplotlib
- seaborn

**Daten einlesen "Loan_Data.csv"**


**Explorative Datenanalyse:**

In diesem Teil wird die explorative Datenanalyse ausgeführt. Es werden unterschiedliche Diagramme generiert, um Zusammenhänge und Muster zu entdecken. Ziel dieses Abschnitts ist es, durch verschiedene Aufgaben ein umfassendes Verständnis der Daten zu erlangen. Dabei sollen wesentliche Trends und Muster aufgedeckt werden, die zur Vorhersage der Rückzahlungsfähigkeit von Kreditnehmern beitragen können:

- Histogramm der FICO-Verteilungen: Erstellt ein Histogramm, das zwei FICO-Verteilungen übereinander darstellt, basierend auf den Ergebnissen der “credit.policy”.
- Histogramm nach “not.fully.paid”: Erstellt ein ähnliches Histogramm wie oben, aber diesmal trennt es die Daten basierend auf der “not.fully.paid” Spalte.
- Countplot der Darlehen nach Zweck: Erstellt ein Countplot mit Seaborn, das die Anzahl der Darlehen nach ihrem Zweck darstellt.
- Jointplot von FICO Score und Zinsen: Untersucht den Trend zwischen dem FICO Score und den Zinsen durch die Erstellung eines Jointplots.
- Lmplots für “not.fully.paid” und “credit.policy”: Erstellt ein lmplots, um zu sehen, ob sich der Trend zwischen “not.fully.paid” und “credit.policy” unterscheidet. Teilt im Anschluss die Daten in zwei Spalten auf.

**Daten vorbereiten:**

In diesem Abschnitt werden Umformungen durchgeführt, um die Daten für die weitere Analyse vorzubereiten. Dies beinhaltet die Umwandlung kategorischer Merkmale in numerische Werte und die Speicherung der transformierten Daten in einem neuen DataFrame.

**Train Test Split:**

In diesem Abschnitt wird der Datensatz in Trainings- und Testsets aufgeteilt. Mit Hilfe der `train_test_split` Funktion aus der `sklearn` Bibliothek werden die unabhängigen Variablen (X) von der Zielvariable ('not.fully.paid') getrennt und dann in Trainings- und Testsets unterteilt. Dies ermöglicht eine unabhängige Validierung des Modells auf dem Testset nach dem Training auf dem Trainingsset.

**Entscheidungsbaummodell trainieren:**

In diesem Abschnitt wird ein Entscheidungsbaum-Modell erstellt und trainiert. Anschließend wird eine Instanz des DecisionTreeClassifier namens dtree erstellt. Schließlich wird das Modell mit den Trainingsdaten trainiert, indem die fit Methode auf dtree angewendet wird. Dieser Prozess passt das Modell an die Trainingsdaten an, um Muster zu erkennen, die zur Vorhersage der Zielvariable verwendet werden können.

**Vorhersagen und Auswertung:**

Das Entscheidungsbaum-Modell hat eine Genauigkeit von 73%. Es ist gut darin, Klasse 0 vorherzusagen, aber nicht bei  derKlasse 1. Die Confusion Matrix zeigt mehr falsche Vorhersagen für Klasse 1. Verbesserungen könnten erforderlich sein, um die Vorhersageleistung für Klasse 1 zu erhöhen.

**Random Forest Modell trainieren:**

In diesem Abschnitt wird ein Random Forest Modell erstellt und trainiert.

**Vorhersagen und Auswertung:**

Das Random Forest Modell hat eine Genauigkeit von 85%. Es ist sehr gut darin, Klasse 0 vorherzusagen, aber es hat Schwierigkeiten bei der Vorhersage von Klasse 1. Die Confusion Matrix zeigt, dass das Modell die meisten Klasse 1 Fälle falsch vorhersagt. Es könnte notwendig sein, das Modell zu verbessern, um die Vorhersageleistung für Klasse 1 zu erhöhen.
