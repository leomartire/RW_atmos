       character function char_int(nm)
c      input an interger and return the last 4 digits as a character
c      character char_int*4

       n=nm
       do 10 i=4,1,-1
       m=int(n/10)
       id=n-10*m
       n=m
  10   char_int(i:i)=char(id+48)

       return
       end
