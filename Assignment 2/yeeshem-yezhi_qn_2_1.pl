% facts
monarch(elizabeth).
son(elizabeth, charles, 1).
son(elizabeth, andrew, 3).
son(elizabeth, edward, 4).
daughter(elizabeth, ann, 2).

% Relational rule: X precedes Y under old succession
old_precedes(X, Y) :-
    son(elizabeth, X, _),
    daughter(elizabeth, Y, _).

old_precedes(X, Y) :-
    son(elizabeth, X, Bx),
    son(elizabeth, Y, By),
    Bx < By.

old_precedes(X, Y) :-
    daughter(elizabeth, X, Bx),
    daughter(elizabeth, Y, By),
    Bx < By.

succession(M) :-
    monarch(M),
    setof(BirthOrder-Child, son(M, Child, BirthOrder), Sons),
    setof(BirthOrder-Child, daughter(M, Child, BirthOrder), Daughters),
    append(Sons, Daughters, Line),
    print_line(Line).

print_line([]).
print_line([_-Child|Tail]) :-
    writeln(Child),
    print_line(Tail).