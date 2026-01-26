const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  console.log('=== YouTube切り抜き動画生成ツール テスト ===\n');

  // テストする動画リスト
  const videos = [
    { name: 'YOASOBI - アイドル', url: 'https://www.youtube.com/watch?v=ZRtdQ81jPUQ' },
    { name: 'Official髭男dism - Pretender', url: 'https://www.youtube.com/watch?v=TQ8WlA2GXbk' },
  ];

  await page.goto('https://youtube-cut-lilac.vercel.app');
  console.log('ページにアクセスしました\n');

  for (let i = 0; i < videos.length; i++) {
    const video = videos[i];
    console.log(`--- テスト ${i + 1}: ${video.name} ---`);

    // 入力欄をクリア
    await page.fill('input', '');

    // URL入力
    await page.fill('input', video.url);
    console.log(`URL入力: ${video.url}`);

    // ボタンクリック
    await page.click('button');
    console.log('検出ボタンをクリック');

    // 結果を待つ
    await page.waitForTimeout(5000);

    // 結果取得
    const resultText = await page.$eval('#result', el => el.innerText).catch(() => '結果なし');
    console.log('結果:\n' + resultText);

    // スクリーンショット
    await page.screenshot({ path: `test_video_${i + 1}.png`, fullPage: true });
    console.log(`スクリーンショット保存: test_video_${i + 1}.png\n`);

    await page.waitForTimeout(1000);
  }

  await browser.close();
  console.log('✅ 全テスト完了');
})();
