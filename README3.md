好
基本上我只有把color quantization table的最右下角調成32767;
因為我其他的壓縮率或保品質嘗試皆以失敗告終
我試了包括
1.clamp時, 把-128 to +127的 range先壓小, 然後在decodeer在用一些rule baesd的方式還原
但結果是, 壓縮率不減反增

2.我做了類似跳取的嘗試 -128 -127 用-128代替, -126 -125用-126代替, etc... 但也沒屁用

3.用6 or 7 bits來quantization --> 壓縮率比前面還爛