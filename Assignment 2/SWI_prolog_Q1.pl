competitor(sumsum, appy).
developer(sumsum, galactica_s3).
smart_phone_technology(galactica_s3).
boss(stevey, appy).
stole(stevey, galactica_s3).

unethical(X) :- boss(X, Y), stole(X, Z), business(Z), rival(A, Y), developer(A, Z).
rival(X, Y) :- competitor(X, Y).
business(X) :- smart_phone_technology(X).
