#ifndef PARSER_H
#define PARSER_H

typedef enum {
  CLA_TESTING,
  CLA_TRAINING,
  CLA_GUI,
} CommandLineAction;

typedef struct {
  CommandLineAction action;
  const char *data_path;
} CommandLineArgs;

void command_line_parse(CommandLineArgs *command_line, int argc, char *argv[]);

#endif // PARSER_H
