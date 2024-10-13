SRC_DIR := ./src
INC_DIR := ./src
LIB_DIR := ./lib
TEST_DIR := ./tests

RELEASE := 0

BUILD_DIR := ./build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

_EXCLUDE :=
EXCLUDE  := $(_EXCLUDE:%=$(SRC_DIR)/%) # Prepending SRC_DIR path

EXE := $(BIN_DIR)/ditect
SRC := $(filter-out $(EXCLUDE), $(wildcard $(SRC_DIR)/*.c))
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TESTS := $(wildcard $(TEST_DIR)/*.c)
TEST_EXE := $(TESTS:$(TEST_DIR)/%.c=$(BIN_DIR)/%)

# If RELEASE var is set to 1
ifeq ($(RELEASE),1)
	OPTFLAG := -Ofast
else
	OPTFLAG := -O0 -ggdb
endif

ifeq ("$(shell uname -m)","x86_64")
	C_RAYLIB=-I ./raylib-5.0_linux_amd64/include/ 
	LD_RAYLIB=-L./raylib-5.0_linux_amd64/lib -l:libraylib.a
else
	C_RAYLIB=$(shell pkg-config --cflags "raylib")
	LD_RAYLIB=$(shell pkg-config --libs "raylib")
endif

CPPFLAGS := -I$(INC_DIR) -MMD -MP
CFLAGS   := -Wall -Werror -Wextra -Wpedantic $(OPTFLAG) $(C_RAYLIB)
LDFLAGS  := -L$(LIB_DIR) $(OPTFLAG)
LDLIBS   := -lm $(LD_RAYLIB)

define DEPENDABLE_VAR
.PHONY: phony
$(BUILD_DIR)/$1: phony | $(BUILD_DIR)
	@if [ "$(shell cat $(BUILD_DIR)/$1 2>&1)" != "$($1)" ]; then \
		echo -n $($1) > $(BUILD_DIR)/$1 ; \
	fi
endef

.PHONY: all
all: $(EXE)
	@echo 'Run "$^" to start game.'

# Make RELEASE depedable
$(eval $(call DEPENDABLE_VAR,RELEASE))

.PHONY: run
run: $(EXE)
	@./$^

.PHONY: clean
clean:
	@$(RM) -rv $(BUILD_DIR)

# Linking:
$(EXE): $(OBJ) | $(BIN_DIR)
	@echo "Linking..."
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Compiling:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(BUILD_DIR)/RELEASE | $(OBJ_DIR)
	@echo "Compiling..."
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Creat dirs:
$(BIN_DIR) $(OBJ_DIR) $(BUILD_DIR):
	mkdir -p $@

COMPILE_DB := compile_flags.txt

.PHONY: compiledb
compiledb:
	compiledb make all tests

$(BIN_DIR)/%: $(TEST_DIR)/%.c $(BUILD_DIR)/RELEASE | $(BIN_DIR)
	@echo "Compiling tests..."
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: tests
tests: $(TEST_EXE)

.PHONY: run-tests
run-tests: tests
	@for exe in "$(TEST_EXE)"; do \
		./$$exe; \
	done

-include $(OBJ:.o=.d)
-include $(TEST_EXE:=.d)
