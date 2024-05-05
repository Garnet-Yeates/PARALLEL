FCC = mpicc
LDC = mpicc
LD_FLAGS = -std=c11
FLAGS = -std=c11
PROG = PA2.cx
RM = /bin/rm
OBJS = PA2.o

#all rule
all: $(PROG)

$(PROG): $(OBJS)
	$(LDC) $(LD_FLAGS) $(OBJS) -o $(PROG)

%.o: %.c
	$(FCC) $(FLAGS) -c $<

#clean rule
clean:
	$(RM) -rf *.o $(PROG) *.mod
