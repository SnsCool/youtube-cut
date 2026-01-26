const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  console.log('1. アクセス中: https://youtube-cut-lilac.vercel.app');
  await page.goto('https://youtube-cut-lilac.vercel.app');

  console.log('2. ページタイトル:', await page.title());

  // スクリーンショット（初期状態）
  await page.screenshot({ path: 'screenshot_1_initial.png', fullPage: true });
  console.log('3. 初期画面のスクリーンショット保存: screenshot_1_initial.png');

  // YouTube URLを入力
  const testUrl = 'https://www.youtube.com/watch?v=9bZkp7q19f0';
  console.log('4. URL入力:', testUrl);
  await page.fill('input', testUrl);

  // スクリーンショット（入力後）
  await page.screenshot({ path: 'screenshot_2_input.png', fullPage: true });
  console.log('5. URL入力後のスクリーンショット保存: screenshot_2_input.png');

  // ボタンをクリック
  console.log('6. 「盛り上がりポイントを検出」ボタンをクリック');
  await page.click('button');

  // 結果を待つ（最大30秒）
  console.log('7. 結果を待機中...');
  try {
    await page.waitForSelector('#result li, #result .error', { timeout: 30000 });
    console.log('8. 結果が表示されました');
  } catch (e) {
    console.log('8. タイムアウト - 結果が表示されませんでした');
  }

  // スクリーンショット（結果）
  await page.screenshot({ path: 'screenshot_3_result.png', fullPage: true });
  console.log('9. 結果画面のスクリーンショット保存: screenshot_3_result.png');

  // 結果のテキストを取得
  const resultText = await page.$eval('#result', el => el.innerText).catch(() => 'No result found');
  console.log('10. 検出結果:\n' + resultText);

  await browser.close();
  console.log('\n✅ テスト完了');
})();
