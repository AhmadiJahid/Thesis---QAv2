Classify the question complexity. Output: 1, 2, or 3

Q: What genre is Inception?
Trace: Inception → genre
A: 1

Q: Who directed Titanic?
Trace: Titanic → director
A: 1

Q: What movies did Tom Hanks act in?
Trace: Tom Hanks → movies
A: 1

Q: Who directed movies starring Brad Pitt?
Trace: Brad Pitt → movies → directors
A: 2

: What genres are films directed by Nolan?
Trace: Nolan → films → genres
A: 2

: What release years are movies starring Emma Watson?
Trace: Emma Watson → movies → release years
A: 2

: The director of Avatar directed which other movies?
Trace: Avatar → director → other movies
A: 2

: What languages are films that share directors with The Matrix?
Trace: The Matrix → director → other films → languages
A: 3

: Who acted in movies whose directors also directed Inception?
Trace: Inception → director → other movies → actors
A: 3


Q: {question}
Trace:
