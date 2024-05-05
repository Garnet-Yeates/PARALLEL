FCC = gcc
LDC = gcc
LD_FLAGS = -lm -fopenmp
FLAGS = -fopenmp
PROG = LA3.cx
OBJS = LA3.o
RM = /bin/rm

#all rule
all: $(PROG)

$(PROG): $(OBJS)
	$(LDC) $(LD_FLAGS) $(OBJS) -o $(PROG)

%.o: %.c
	$(FCC) $(FLAGS) -c $<

#clean rule
clean:
	$(RM) -rf *.o $(PROG) *.mod
