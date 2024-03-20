#!/usr/bin/perl

use strict;
use warnings;

#####################################################################################
#                                  Main program body                                #
#####################################################################################
my %backboneAtom = (
    "C"  => 1,
    "O" => 1,
    "N"  => 1,
    "CA" => 1,
    "C" => 1,
);
&RunScript;

#####################################################################################


sub RunScript{
  my ($inputpdb,$ch1,$ch2,$distCutOff,$dif1,$dif2,$beta,$dnds,$dndsFile) = &GetInputOptions(@ARGV);
  my %ch1Coor; #key1: chain_resNumber_resName; #key2: atomName; Value: X_Y_Z
  my %ch2Coor; #key1: chain_resNumber_resName; #key2: atomName; Value: X_Y_Z
  my %ch1Beta; #key1: chain_resNumber_resName; #value: beta value
  my %ch2Beta; #key1: chain_resNumber_resName; #value: beta value
  my %ch1Resnumb2name; #key1: residue number, #value chain_resNumber_resName
  my %ch2Resnumb2name; #key1: residue number, #value chain_resNumber_resName
  &ParsePDB($inputpdb,$ch1,\%ch1Coor,\%ch1Beta,$dif1,\%ch1Resnumb2name);
  &ParsePDB($inputpdb,$ch2,\%ch2Coor,\%ch2Beta,$dif2,\%ch2Resnumb2name);
  my %contact; #key: chain1_resNumber_resName|chain2_resNumber_resName; #value: distance
  #contactType: sidechain-sidechain; backbone-backbone; sidechain-backbone
  #my %contactType; #key: chain1_resNumber_resName|chain2_resNumber_resName; #value: contacttype
  my %contactAtom; #key: chain1_resNumber_resName|chain2_resNumber_resName; #value: atomName1|atomName2
  my (%ch1interf,%ch1interf_minD);
  my (%ch2interf,%ch2interf_minD);
  &GetContacts(\%ch1Coor,\%ch2Coor,\%contact,\%contactAtom,$distCutOff,\%ch1interf,\%ch2interf,\%ch1interf_minD,\%ch2interf_minD);
  
  my $ct = 0;
  print "###############################\n";
  print "Chain $ch1 interfacial residues\n\n";
  #foreach my $interfRes (keys %ch1interf){
  foreach my $resNumber (sort {$a <=> $b} keys %ch1Resnumb2name){
    my $resName = $ch1Resnumb2name{$resNumber};
    next if (!$ch1interf{$resName});
    my $interfRes = $resName;
    my $minD = sprintf("%.1f",$ch1interf_minD{$resName});
    $ct++;
    print "$interfRes\t$ch1interf{$interfRes}";
    print "\t$ch1Beta{$interfRes}" if ($beta eq 'True');
    print "\t$minD angs";
    print "\n";
  }
#=temp
  $ct = 0;
  print "\n###############################\n";
  print "Chain $ch2 interfacial residues\n\n";
  #foreach my $interfRes (keys %ch2interf){
  foreach my $resNumber (sort {$a <=> $b} keys %ch2Resnumb2name){
    my $resName = $ch2Resnumb2name{$resNumber};
    next if (!$ch2interf{$resName});
    my $interfRes = $resName;
    my $minD = sprintf("%.1f",$ch2interf_minD{$resName});
    $ct++;
    print "$interfRes\t$ch2interf{$interfRes}";
    print "\t$ch2Beta{$interfRes}" if ($beta eq "True");
    print "\t$minD angs";
    print "\n";
  }
  $ct = 0;
  print "\n###############################\n";
  print "Contacts within $distCutOff ansgtroms (only showing the closest pair of atoms between residues)\n\n";
  foreach my $intResPair (keys %contact){
    $ct++;
    my $round = sprintf("%.1f",$contact{$intResPair});
    print "$ct\t$intResPair\t$contactAtom{$intResPair}\t$round angs\n";
  }
#=cut
}

#####################################################################################

sub GetContacts{
  my $ch1CoorRef = $_[0];
  my $ch2CoorRef = $_[1];
  my $contactRef = $_[2];
  my $contactAtomRef = $_[3];
  my $distCutOff = $_[4];
  my $ch1interfRef = $_[5];
  my $ch2interfRef = $_[6];
  my $ch1interfMinDRef = $_[7];
  my $ch2interfMinDRef = $_[8];
  
  foreach my $ch1Res (keys %{$ch1CoorRef}){
    foreach my $ch1Atom (keys %{ ${$ch1CoorRef}{$ch1Res} }){
      my ($ch1X,$ch1Y,$ch1Z) = split(/_/,${$ch1CoorRef}{$ch1Res}{$ch1Atom});
      #print "$ch1Res\t$ch1Atom\t$ch1X\t$ch1Y\t$ch1Z\n";
      
      foreach my $ch2Res (keys %{$ch2CoorRef}){
        foreach my $ch2Atom (keys %{ ${$ch2CoorRef}{$ch2Res} }){
          my ($ch2X,$ch2Y,$ch2Z) = split(/_/,${$ch2CoorRef}{$ch2Res}{$ch2Atom});
          
          my $eucDist = &CalculateEucDist($ch1X,$ch1Y,$ch1Z,$ch2X,$ch2Y,$ch2Z);
          next if ($eucDist > $distCutOff);
          my $resKey = $ch1Res . "|" . $ch2Res;
          my $atomPair = $ch1Atom . "|" . $ch2Atom;
          my ($ch1AtomType,$ch2AtomType,$contactType) = &GetContactType($ch1Atom,$ch2Atom);
          ${$ch1interfRef}{$ch1Res} = $ch1AtomType if (!${$ch1interfRef}{$ch1Res});
          ${$ch2interfRef}{$ch2Res} = $ch2AtomType if (!${$ch2interfRef}{$ch2Res});
          ${$ch1interfRef}{$ch1Res} .= "," . $ch1AtomType if (${$ch1interfRef}{$ch1Res} !~m/$ch1AtomType/);
          ${$ch2interfRef}{$ch2Res} .= "," . $ch2AtomType if (${$ch2interfRef}{$ch2Res} !~m/$ch2AtomType/);
          
          if (not defined ${$ch1interfMinDRef}{$ch1Res}){
            ${$ch1interfMinDRef}{$ch1Res} = $eucDist 
          }elsif ($eucDist < ${$ch1interfMinDRef}{$ch1Res}){
            ${$ch1interfMinDRef}{$ch1Res} = $eucDist 
          }
          if (not defined ${$ch2interfMinDRef}{$ch2Res}){
            ${$ch2interfMinDRef}{$ch2Res} = $eucDist 
          }elsif ($eucDist < ${$ch2interfMinDRef}{$ch2Res}){
            ${$ch2interfMinDRef}{$ch2Res} = $eucDist 
          }
          
          if (!${$contactRef}{$resKey}){ 
            ${$contactRef}{$resKey} = $eucDist;
            ${$contactAtomRef}{$resKey} = $atomPair . "\t" . $contactType;
          }elsif ($eucDist < ${$contactRef}{$resKey}){
            ${$contactRef}{$resKey} = $eucDist;
            ${$contactAtomRef}{$resKey} = $atomPair . "\t" . $contactType;
          }elsif ($eucDist == ${$contactRef}{$resKey}){
            ${$contactAtomRef}{$resKey} .= "; " . $atomPair . "\t" . $contactType;
          }
        }
      }
    }
  }
}

#####################################################################################

sub GetContactType{
  my $ch1Atom = $_[0];
  my $ch2Atom = $_[1];
  
  if ( ($backboneAtom{$ch1Atom}) && ($backboneAtom{$ch2Atom}) ){
    return("backbone","backbone","backbone-backbone");
  }elsif ( (!$backboneAtom{$ch1Atom}) && (!$backboneAtom{$ch2Atom}) ){
    return("sidechain","sidechain","sidechain-sidechain");
  }elsif ( ($backboneAtom{$ch1Atom}) && (!$backboneAtom{$ch2Atom}) ){
    return("backbone","sidechain","backbone-sidechain");
  }elsif ( (!$backboneAtom{$ch1Atom}) && ($backboneAtom{$ch2Atom}) ){
    return("sidechain","backbone","sidechain-backbone");
  }
}

#####################################################################################

sub CalculateEucDist{
  my $ch1X = $_[0];
  my $ch1Y = $_[1];
  my $ch1Z = $_[2];
  my $ch2X = $_[3];
  my $ch2Y = $_[4];
  my $ch2Z = $_[5];
  
  my $difx = $ch1X - $ch2X;
  my $difxSq = $difx**2;
  my $dify = $ch1Y - $ch2Y;
  my $difySq = $dify**2;
  my $difz = $ch1Z - $ch2Z;
  my $difzSq = $difz**2;
  my $sum = $difxSq + $difySq + $difzSq;
  my $eucDist = sqrt($sum);
  
  return($eucDist);
}
#####################################################################################

sub ParsePDB{
  my $file = $_[0];
  my $ch = $_[1];
  my $hashRef = $_[2];
  my $betaHashRef = $_[3];
  my $dif = $_[4];
  my $Num2NameRef = $_[5];

  open(IN,"<",$file) or die "cant open $file\n";
  
  while(<IN>){
    my $s = $_;
    chomp($s);
    next if (length($s)<10);
    next if (substr($s,0,4) ne "ATOM");
    my $keyword = substr($s,0,4);
    my $atomName = substr($s,12,4);
    $atomName =~ s/\s//g;
    my $resName = substr($s,17,3);
    my $resNumber = substr($s,22,4);
    $resNumber =~ s/\s//g;
    $resNumber = $resNumber + $dif;
    my $chain = substr($s,21,1);
    if ( ($keyword eq 'ATOM') && ($chain eq $ch) ){
      #print "$s\n";
      my $xCoor = substr($s,30,8);
      $xCoor =~ s/\s//g;
      my $yCoor = substr($s,38,8);
      $yCoor =~ s/\s//g;
      my $zCoor = substr($s,47,8);
      $zCoor =~ s/\s//g;
      #print "$chain\_$resName\_$resNumber\t$atomName\t$xCoor  $yCoor  $zCoor\n";
      ${$hashRef}{"$ch\_$resName\_$resNumber"}{$atomName} = "$xCoor\_$yCoor\_$zCoor";
      my $beta = substr($s,60,6);
      $beta =~ s/\s//g;
      ${$betaHashRef}{"$ch\_$resName\_$resNumber"} = $beta if (!${$betaHashRef}{"$ch\_$resName\_$resNumber"});
      ${$Num2NameRef}{$resNumber} = "$ch\_$resName\_$resNumber" if (!${$Num2NameRef}{$resNumber});
    }
  }
  close IN;

}

#####################################################################################

sub GetInputOptions{
    my @arguments = @_;
    my $PGM;
    my $usage;
    my $s;
    my ($inputpdb,$chain1,$chain2,$distCutOff,$mode);
    my $dif1 = 0;
    my $dif2 = 0;
    my $beta = 'False';
    my $dnds = 'False';
    my $dndsFile = 'null';
    
    $PGM = $0;
    $PGM =~ s#.*/##;                #remove part up to last slash
    
    $usage = <<USAGE;
    Usage:
$PGM -i input pdb -c1 chain 1 -c2 chain 2 -d distance cutoff -dif1 370 -dif2 0 -beta True -dnds ../DnDs/NEB_AllBat_dNdS.csv
    options:
        e.g. $PGM -i 5f1b.pdb -c1 A -c2 C -d 5 -dif1 0 -dif2 372
        dif1 and dif2 are optional, corrects the residue numbering according to a
          user-defined value
USAGE

    while(@arguments){
        $s = shift(@arguments);
        if ($s){
                if ($s eq "-i") {$inputpdb = shift(@arguments); next;}
                if ($s eq "-c1") {$chain1 = shift(@arguments); next;}
                if ($s eq "-c2") {$chain2 = shift(@arguments); next;}
                if ($s eq "-d") {$distCutOff = shift(@arguments); next;}
                if ($s eq "-dif1") {$dif1 = shift(@arguments); next;}
                if ($s eq "-dif2") {$dif2 = shift(@arguments); next;}
                if ($s eq "-beta") {$beta = shift(@arguments); next;}
                if ($s eq "-dnds") {$dnds = 'True'; $dndsFile = shift(@arguments); next;}
            }
    }

    if ((!$inputpdb) || (!$chain1) || (!$chain2) || (!$distCutOff)) {&PrintCommandError("error: input params not defined",$usage)}
    
    return ($inputpdb,$chain1,$chain2,$distCutOff,$dif1,$dif2,$beta,$dnds,$dndsFile);
}

#####################################################################################

sub PrintCommandError{
    my $errorMessage = $_[0];
    my $usage = $_[1];
    
    print "\n". $errorMessage . "\n";
    print "\n". $usage . "\n";
    exit;
}

#####################################################################################