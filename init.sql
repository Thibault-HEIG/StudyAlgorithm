CREATE TABLE grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    final_grade INTEGER NOT NULL
);

INSERT INTO grades (final_grade) VALUES (100);

SELECT * FROM grades;