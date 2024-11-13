#include "parser.h"
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void command_line_parse(CommandLineArgs *command_line, int argc, char *argv[]) {
  CommandLineAction action = CLA_GUI;
  char *data_path = NULL;
  const char err[] = "%s: Either specify testing or training, not both!\n";

  while (1) {
    static struct option long_options[] = {{"train", required_argument, 0, 't'},
                                           {"test", required_argument, 0, 'T'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    int c = getopt_long(argc, argv, "t:T:h", long_options, &option_index);

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

    case 'h':
      printf("Usage: %s [OPTION]...\n\n", argv[0]);
      printf(
          "This program predicts handwritten digits using a neural network.\n"
          "With no options, the program will start a GUI, where you can draw a "
          "digit and the network tries to predict it.\n"
          "It is also possible to train and test the network using the "
          "different OPTIONS.\n\n");
      printf("  -t, --train=FILE    Train the network with the data in "
             "FILE\n");
      printf("  -T, --test=FILE     Test the network with the data in "
             "FILE\n");
      printf("  -h, --help          Display this help and exit\n");
      exit(0);

    case '?':
      /* getopt_long already printed an error message. */
      exit(1);

    default:
      abort();
    }
  }

  command_line->action = action;

  if (data_path) {
    const size_t len = strlen(data_path);
    if (data_path[len - 1] == '/') {
      data_path[len - 1] = '\0';
    }
  }
  command_line->data_path = data_path;
}
