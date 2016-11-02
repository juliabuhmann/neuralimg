#!/usr/bin/python

from subprocess import check_call, Popen, PIPE


class Process(object):

    def __init__(self, cmd, log=None, out_str=False):
        """ Creates a process to execute in the cmd. Accepts the list of arguments
        and a log file to output the process feedback
        :param cmd: Command to execute
        :param log: Log file where to output results
        :param out_str: Whether to gather the output as a string and return it.
            Otherwise returns always 1"""
        self.cmd = cmd
        self.log_file = log
        self.out_str = out_str

    def execute(self, verbose=False):
        """ Executes the command and stores it in the corresponding steam.
        If """

        if verbose is True:
            print("Running: " + ' '.join(self.cmd))

        if self.log_file is None and self.out_str is False:
            check_call(self.cmd)
        else:
            # Call, wait for end and store results in log
            self.proc = Popen(self.cmd, stdout=PIPE, stderr=PIPE)
            #self.proc.wait() # This is commented, seems to cause problems in CRAG generation

            if self.log_file is not None:
                self._store_output()
            if self.out_str is True:
                return self._gather_string()

        return 1

    def _gather_string(self):
        """ Stores output in the log file """
        return _serialize_out(self.proc.stdout) + _serialize_out(self.proc.stderr)

    def _store_output(self):
        """ Stores output in the log file """
        with open(self.log_file, 'w') as w:
            while self.proc.poll() is None:
                _gather_out(w, self.proc.stdout)
                _gather_out(w, self.proc.stderr)


def _serialize_out(pipe, verbose=True):
    """ Returns the output of the channel as a string """
    line = pipe.readline()
    res = ''
    while line:
        if verbose is True:
            print(line.strip())
        res = res + str(line)
        line = pipe.readline()
    return res


def _gather_out(f, pipe, verbose=True):
    """ Writes the output of the channel into the file """
    line = pipe.readline()
    while line:
        if verbose is True:
            print(line.strip())
        f.write(str(line))
        line = pipe.readline()

