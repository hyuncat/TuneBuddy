% Lily was here -- automatically converted by midi2ly from fugue.mid
\version "2.14.0"

\layout {
  \context {
    \Voice
    \remove Note_heads_engraver
    \consists Completion_heads_engraver
    \remove Rest_engraver
    \consists Completion_rest_engraver
  }
}

trackAchannelA = {


  \key c \major
    
  \tempo 4 = 80 
  

  \key c^\markup { "↑" } \major
  
  \time 4/4 
  \skip 4 
}

trackA = <<
  \context Voice = voiceA \trackAchannelA
>>


trackBchannelA = {
  
  \set Staff.instrumentName = "Violin"
  \skip 2*9 
}

trackBchannelB = \relative c {
  d'16^\markup { "↑" } a' cis e^\markup { "↑" } f a d, c b d f gis g f dis d 
  | % 2
  dis g, c d dis g c, ais a c dis g f dis d c 
  | % 3
  d a ais fis g ais d f, e g ais d c a ais g 
  | % 4
  fis a c dis d c ais a ais g d' f, dis d' g c, 
  | % 5
  fis4 
}

trackB = <<
  \context Voice = voiceA \trackBchannelA
  \context Voice = voiceB \trackBchannelB
>>


\score {
  <<
    \context Staff=trackB \trackA
    \context Staff=trackB \trackB
  >>
  \layout {}
  \midi {}
}
