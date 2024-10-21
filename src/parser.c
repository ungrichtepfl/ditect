#include "parser.h"
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void command_line_parse(CommandLineArgs *command_line, int argc, char *argv[]) {
  CommandLineAction action = CLA_GUI;
  const char *data_path = NULL;
  const char err[] = "%s: Either specify testing or training, not both!\n";

  while (1) {
    static struct option long_options[] = {{"train", required_argument, 0, 't'},
                                           {"test", required_argument, 0, 'T'},
                                           {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    int c = getopt_long(argc, argv, "t:T:", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 't':
      if (data_path) {
        fprintf(stderr, err, argv[0]);
        exit(1);
      }
      action = CLA_TRAINING;
      data_path = optarg;
      break;

    case 'T':
      if (data_path) {
        fprintf(stderr, err, argv[0]);
        exit(1);
      }
      action = CLA_TESTING;
      data_path = optarg;
      break;

    case '?':
      /* getopt_long already printed an error message. */
      exit(1);

    default:
      abort();
    }
  }

  command_line->action = action;
  command_line->data_path = data_path;
}
