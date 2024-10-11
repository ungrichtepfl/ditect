SRC_DIR := src
INC_DIR := src
LIB_DIR := lib

RELEASE := 0

BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

_EXCLUDE :=
EXCLUDE  := $(_EXCLUDE:%=$(SRC_DIR)/%) # Prepending SRC_DIR path

EXE := $(BIN_DIR)/ditect
SRC := $(filter-out $(EXCLUDE), $(wildcard $(SRC_DIR)/*.c))
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# If RELEASE var is set to 1
ifeq ($(RELEASE),1)
	OPTFLAG := -Ofast
else
	OPTFLAG := -O0 -ggdb
endif

ifeq ($(shell uname -m),x86_64 )
	RAYLIB=-I ./raylib-5.0_linux_amd64/include/ -L./raylib-5.0_linux_amd64/lib -l:libraylib.a -ldl
else
	RAYLIB=$(shell pkg-config --libs --cflags "raylib")
endif

CPPFLAGS := -I$(INC_DIR) -MMD -MP
CFLAGS   := -Wall -Werror -Wextra -Wpedantic $(OPTFLAG)
LDFLAGS  := -L$(LIB_DIR) $(OPTFLAG)
SDL2LIB  := `sdl2-config --cflags --libs` -lSDL2_ttf
LDLIBS   := -lm $(RAYLIB)

define DEPENDABLE_VAR
.PHONY: phony
$(BUILD_DIR)/$1: phony
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
$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

-include $(OBJ:.o=.d)
