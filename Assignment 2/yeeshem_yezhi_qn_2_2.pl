% facts
monarch(elizabeth).
son(elizabeth, charles, 1).
son(elizabeth, andrew, 3).
son(elizabeth, edward, 4).
daughter(elizabeth, ann, 2).


child(M, Child, BirthOrder) :- son(M, Child, BirthOrder).
child(M, Child, BirthOrder) :- daughter(M, Child, BirthOrder).

% new succession rule
new_succession(M) :-
    monarch(M),
    setof(BirthOrder-Child, child(M, Child, BirthOrder), Line),
    print_line(Line).


print_line([]).
print_line([_-Child|Tail]) :-
    writeln(Child),
    print_line(Tail).