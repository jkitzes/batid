FasdUAS 1.101.10   ��   ��    k             l     ��  ��    8 2 Short Applescript to launch arbitrary python file     � 	 	 d   S h o r t   A p p l e s c r i p t   t o   l a u n c h   a r b i t r a r y   p y t h o n   f i l e   
  
 l     ��  ��    0 * Believed to work on 10.6+, tested on 10.8     �   T   B e l i e v e d   t o   w o r k   o n   1 0 . 6 + ,   t e s t e d   o n   1 0 . 8      l     ��  ��    * $ Justin Kitzes, jkitzes@berkeley.edu     �   H   J u s t i n   K i t z e s ,   j k i t z e s @ b e r k e l e y . e d u      l     ��������  ��  ��        l     ��  ��    G A Save executable in AppleScript Editor with Export -> Application     �   �   S a v e   e x e c u t a b l e   i n   A p p l e S c r i p t   E d i t o r   w i t h   E x p o r t   - >   A p p l i c a t i o n      l     ��������  ��  ��        l     ����  r        !   m      " " � # #  R u n _ B a t I D . p y ! o      ���� 
0 pyfile  ��  ��     $ % $ l     ��������  ��  ��   %  & ' & l     �� ( )��   ( %  Check if Terminal already open    ) � * * >   C h e c k   i f   T e r m i n a l   a l r e a d y   o p e n '  + , + l    -���� - O    . / . r     0 1 0 l    2���� 2 I   �� 3��
�� .coredoexbool       obj  3 4    �� 4
�� 
prcs 4 m   
  5 5 � 6 6  T e r m i n a l��  ��  ��   1 o      ���� 0 
terminalon 
terminalOn / m     7 7�                                                                                  sevs  alis    �  Mountain Lion              �Xo�H+     0System Events.app                                               0b��Y        ����  	                CoreServices    �X�j      ���       0   *   )  >Mountain Lion:System: Library: CoreServices: System Events.app  $  S y s t e m   E v e n t s . a p p    M o u n t a i n   L i o n  -System/Library/CoreServices/System Events.app   / ��  ��  ��   ,  8 9 8 l     ��������  ��  ��   9  : ; : l     �� < =��   < , & Open new window or run in same window    = � > > L   O p e n   n e w   w i n d o w   o r   r u n   i n   s a m e   w i n d o w ;  ? @ ? l     �� A B��   A S M Repetition appears necessary to avoid opening a second blank Terminal window    B � C C �   R e p e t i t i o n   a p p e a r s   n e c e s s a r y   t o   a v o i d   o p e n i n g   a   s e c o n d   b l a n k   T e r m i n a l   w i n d o w @  D E D l   � F���� F Z    � G H�� I G l    J���� J o    ���� 0 
terminalon 
terminalOn��  ��   H k    N K K  L M L O   ) N O N e    ( P P n    ( Q R Q 1   % '��
�� 
psxp R l   % S���� S c    % T U T l   # V���� V n    # W X W m   ! #��
�� 
ctnr X l   ! Y���� Y I   !�� Z��
�� .earsffdralis        afdr Z  f    ��  ��  ��  ��  ��   U m   # $��
�� 
ctxt��  ��   O m     [ [�                                                                                  MACS  alis    x  Mountain Lion              �Xo�H+     0
Finder.app                                                      �4�k"        ����  	                CoreServices    �X�j      �͒       0   *   )  7Mountain Lion:System: Library: CoreServices: Finder.app    
 F i n d e r . a p p    M o u n t a i n   L i o n  &System/Library/CoreServices/Finder.app  / ��   M  \ ] \ O  * 6 ^ _ ^ I  . 5�� `��
�� .coredoscnull��� ��� ctxt ` b   . 1 a b a m   . / c c � d d  c d   b 1   / 0��
�� 
rslt��   _ m   * + e e�                                                                                      @ alis    p  Mountain Lion              �Xo�H+     NTerminal.app                                                     $���g        ����  	                	Utilities     �X�j      ���       N   M  3Mountain Lion:Applications: Utilities: Terminal.app     T e r m i n a l . a p p    M o u n t a i n   L i o n  #Applications/Utilities/Terminal.app   / ��   ]  f�� f O  7 N g h g I  ; M�� i j
�� .coredoscnull��� ��� ctxt i b   ; @ k l k m   ; > m m � n n  p y t h o n   l o   > ?���� 
0 pyfile   j �� o��
�� 
kfil o 4  C I�� p
�� 
cwin p m   G H���� ��   h m   7 8 q q�                                                                                      @ alis    p  Mountain Lion              �Xo�H+     NTerminal.app                                                     $���g        ����  	                	Utilities     �X�j      ���       N   M  3Mountain Lion:Applications: Utilities: Terminal.app     T e r m i n a l . a p p    M o u n t a i n   L i o n  #Applications/Utilities/Terminal.app   / ��  ��  ��   I k   Q � r r  s t s O  Q b u v u e   U a w w n   U a x y x 1   ^ `��
�� 
psxp y l  U ^ z���� z c   U ^ { | { l  U \ }���� } n   U \ ~  ~ m   Z \��
�� 
ctnr  l  U Z ����� � I  U Z�� ���
�� .earsffdralis        afdr �  f   U V��  ��  ��  ��  ��   | m   \ ]��
�� 
ctxt��  ��   v m   Q R � ��                                                                                  MACS  alis    x  Mountain Lion              �Xo�H+     0
Finder.app                                                      �4�k"        ����  	                CoreServices    �X�j      �͒       0   *   )  7Mountain Lion:System: Library: CoreServices: Finder.app    
 F i n d e r . a p p    M o u n t a i n   L i o n  &System/Library/CoreServices/Finder.app  / ��   t  � � � O  c z � � � I  g y�� � �
�� .coredoscnull��� ��� ctxt � b   g l � � � m   g j � � � � �  c d   � 1   j k��
�� 
rslt � �� ���
�� 
kfil � 4  o u�� �
�� 
cwin � m   s t���� ��   � m   c d � ��                                                                                      @ alis    p  Mountain Lion              �Xo�H+     NTerminal.app                                                     $���g        ����  	                	Utilities     �X�j      ���       N   M  3Mountain Lion:Applications: Utilities: Terminal.app     T e r m i n a l . a p p    M o u n t a i n   L i o n  #Applications/Utilities/Terminal.app   / ��   �  ��� � O  { � � � � I   ��� � �
�� .coredoscnull��� ��� ctxt � b    � � � � m    � � � � � �  p y t h o n   � o   � ����� 
0 pyfile   � �� ���
�� 
kfil � 4  � ��� �
�� 
cwin � m   � ����� ��   � m   { | � ��                                                                                      @ alis    p  Mountain Lion              �Xo�H+     NTerminal.app                                                     $���g        ����  	                	Utilities     �X�j      ���       N   M  3Mountain Lion:Applications: Utilities: Terminal.app     T e r m i n a l . a p p    M o u n t a i n   L i o n  #Applications/Utilities/Terminal.app   / ��  ��  ��  ��   E  � � � l     ��������  ��  ��   �  � � � l     �� � ���   � %  Bring Terminal window to front    � � � � >   B r i n g   T e r m i n a l   w i n d o w   t o   f r o n t �  ��� � l  � � ����� � O  � � � � � I  � �������
�� .miscactvnull��� ��� null��  ��   � m   � � � ��                                                                                      @ alis    p  Mountain Lion              �Xo�H+     NTerminal.app                                                     $���g        ����  	                	Utilities     �X�j      ���       N   M  3Mountain Lion:Applications: Utilities: Terminal.app     T e r m i n a l . a p p    M o u n t a i n   L i o n  #Applications/Utilities/Terminal.app   / ��  ��  ��  ��       �� � � "������   � ��������
�� .aevtoappnull  �   � ****�� 
0 pyfile  �� 0 
terminalon 
terminalOn��   � �� ����� � ���
�� .aevtoappnull  �   � **** � k     � � �   � �  + � �  D � �  �����  ��  ��   �   �  "�� 7�� 5���� [�������� e c���� m���� � ����� 
0 pyfile  
�� 
prcs
�� .coredoexbool       obj �� 0 
terminalon 
terminalOn
�� .earsffdralis        afdr
�� 
ctnr
�� 
ctxt
�� 
psxp
�� 
rslt
�� .coredoscnull��� ��� ctxt
�� 
kfil
�� 
cwin
�� .miscactvnull��� ��� null�� ��E�O� *��/j E�UO� ;� )j �,�&�,EUO� 	��%j UO� a �%a *a k/l UY C� )j �,�&�,EUO� a �%a *a k/l UO� a �%a *a k/l UO� *j U
�� boovfals��   ascr  ��ޭ