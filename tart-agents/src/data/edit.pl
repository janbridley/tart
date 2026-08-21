use strict;
use warnings;

my $old = $ENV{TART_OLD};
my $new = $ENV{TART_NEW};
my $all = ($ENV{TART_ALL} // '') eq '1';
my $file = $ARGV[0] // '';
if (!length $file) { warn "edit: no file given\n"; exit 1 }
if (!defined $old || !length $old) { warn "edit: old_string is empty\n"; exit 1 }
if (!defined $new) { warn "edit: new_string is unset\n"; exit 1 }

open(my $fh, '+<', $file) or do { warn "edit: cannot open $file: $!\n"; exit 1 };
binmode($fh);
flock($fh, 2) or do { warn "edit: cannot lock $file: $!\n"; exit 1 };
my $orig = do { local $/; <$fh> };

# Substitute first, discarding the edit if the file has diverged.
my $updated = $orig;
my $n = ($updated =~ s/\Q$old\E/$new/g);
if ($n == 0) { warn "edit: old_string not found in $file\n"; exit 1 }
if ($n > 1 && !$all) { warn "edit: old_string matches $n times in $file\n"; exit 1 }

seek($fh, 0, 0) or do { warn "edit: cannot seek $file: $!\n"; exit 1 };
truncate($fh, 0) or do { warn "edit: cannot truncate $file: $!\n"; exit 1 };
unless (print {$fh} $updated) {
    my $error = $!;
    seek($fh, 0, 0);
    truncate($fh, 0);
    print {$fh} $orig;
    warn "edit: cannot write $file: $error (restore attempted)\n";
    exit 1;
}
close($fh) or do { warn "edit: cannot close $file: $! (file may be damaged)\n"; exit 1 };
print "edited $file: $n replacement(s)\n";
