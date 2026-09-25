use warnings;
use Errno qw(ENOENT);
our ($start, $end);

# Number a file's lines `cat -n` style, keeping only the bounded lines. The
# The harness prepends a preamble assigning $start and $end, or unbounded if not set.
my $file = $ARGV[0];
open(my $in, '<', $file) or do {
    # Exit 3 names a missing file so the harness can easily catch these cases.
    exit 3 if $! == ENOENT;
    warn "read: cannot open $file: $!\n";
    exit 1
};
flock($in, 1) or do { warn "read: cannot lock $file: $!\n"; exit 1 };
my $lines = 0;
while (<$in>) {
    $lines++;
    printf "%6d\t%s", $., $_ if $. >= ($start || 0);
    exit if $end && $. >= $end;
}
# An empty file gets Claude Code's reminder, since there are no lines to show.
if ($lines == 0) {
    # <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/read-text.ts#L15>
    print "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>\n";
}
