use warnings;

# Number a file's lines `cat -n` style, keeping only the bounded lines. The
# The harness prepends a preamble assigning $start and $end, or unbounded if not set.
my $file = $ARGV[0];
open(my $in, '<', $file) or do { warn "read: cannot open $file: $!\n"; exit 1 };
flock($in, 1) or do { warn "read: cannot lock $file: $!\n"; exit 1 };
while (<$in>) {
    printf "%6d\t%s", $., $_ if $. >= ($start || 0);
    exit if $end && $. >= $end;
}
