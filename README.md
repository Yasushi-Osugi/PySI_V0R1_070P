大杉　泰司(おおすぎ　やすし)

経営コンサルタントとして約20年、主に製造業のグローバル・サプライチェーンの業務課題の解決に取り組んできました。
2020年～2022年のStay homeの２年間に、近年、注目されているAI、機械学習のアプローチを調べるうちに自分でも何か作ってみたくなり、
経営コンサルタントとしての知識、経験をベースとして、AI、機械学習の考え方を参考にして、馴染みの深いグローバル・サプライチェーン向けの在庫発注計画を対象に、機械学習の機能を付加した計画立案ツールのプロトタイプを作成してみましたのでご紹介したいと思います。

【想定される利用者】

想定される利用者は、実務での利用というよりも、大学の経営工学系の学生の教育用に利用いただけるのではないかと思っています。
あるいは、経営コンサルタントのエンゲージメントの中で、サプライチェーン計画のコスト・シミュレーションを行う等々の場面で、コスト・プロファイルを再設定し、評価関数を見直すことで、オペレーション・コストのシミュレーション・ツールとして利用できると思います。

【本計画ツールの特長】

一般的な定期発注方式の在庫計画、PSI計画(PSI:Purchase Sales Inventory)との比較で、ご紹介する計画立案ツールの主な特長は以下のとおりです。
ここで、 この在庫発注計画ツールのプロトタイプをpython言語で開発たPSI計画ツールという意味でPySI(仮称)と呼ぶこととします。
gifファイルPSI_Plan_animation_profit.gifで、PySIが作成した在庫発注計画のアニメーションを見ることができます。

なお、定期発注のタイム・バケットはグローバル・サプライチェーンの実務で一般的な週次バケットのPSI計画として実装しています。
また、本プロトタイプは非常に簡易的なPSI計画機能であることから、実際の現場のデータを入力した場合、例外処理や異常データへの対応は不十分であり、どこまで実用に耐えられるかは未知数です。

1. 従来のPSI計画では、需要と供給のバランスを見ながら、在庫水準を維持することを主な目的として計画立案していますが、ご紹介するPySIでは、PSI計画に機械学習の一つ、Q学習の機能を付加することにより、評価指標(Q学習のreward)として経営視点の利益・売上・利益率などを指定することで、利益最大(コスト最小)、売上最大といった経営目標に合わせたPSI計画を生成することができます。

　例えば、季節商品の商品ライフサイクル全体の利益を最大化したいという場合には、単にシーズン終盤の在庫を絞って売り切るのではなく、利益を最大化する供給量とタイミングを評価しながらPSI計画を生成します。
添付のexcelシートのPySI_monitor_profit.xlsxに利益優先のPSI計画を出力した結果グラフがありますが、これを見ると、利益を評価指標にして利益優先のPSI計画を生成した場合には、シーズン終盤だけでなく、シーズン中盤にも、利益を確保することを優先した結果、需要を充足できずに欠品が発生する PSI計画を生成している様子がわかります。
　一方、売上を優先する場合には、コストと利益を無視してでも売上を上げるためにすべての期間で需要を満たすようにPSI計画(excelシートPySI_monitor_revenue.xlsx参照)を生成します。

2. グローバル・サプライチェーンの物流制約を反映したPSI計画を作成できる。例えば、長期休暇(日本の５月の連休、中国の旧正月など)による出荷・着荷週の制約や、船便の出荷・着荷週の物流制約などを考慮したPSI計画を生成します。

以上が本PSI計画シミュレーション・ツールの特長になります。

なお、将来的な取り組み仮説としては、PySI計画の機能を拡張し、グローバル・サプライチェーン上の各事業単位の間を共通計画単位で連携することにより、統合計画機能を実装することができるとともに、グローバル・オペレーション全体の利益・コスト最適化計画を実行することで、コスト競争力の強化を図ることができると思います。

PySIの機能詳細については、github上に別途、説明資料をuploadしていきたいと思いますが、既に他のSNSサイト(note、はてなブログ)にも「ロット積み上げ方式のPSI計画」といったタイトルで紹介記事を掲載しておりますので、ご興味のある方はそちらもぜひご一読ください。
- 📫 How to reach me : mail ohsugi1031@gmail.com

<!---
Yasushi-Osugi/Yasushi-Osugi is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
