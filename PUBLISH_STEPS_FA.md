# راهنمای تصویری-متنی انتشار CHEM-02 v1.0.0

وضعیت: **فقط بسته MIT آماده شده است؛ هنوز GitHub Release و Zenodo DOI واقعی ساخته نشده‌اند**.

## A — بارگذاری GitHub
1. وارد https://github.com شوید و در صورت نیاز ثبت‌نام کنید.
2. از علامت + بالا، **New repository** را بزنید. نام پیشنهادی: `CHEM-02-solubility-regime-benchmark`.
3. توضیح پیشنهادی: `Reproducible benchmarking of prediction regimes and chemical-domain shift in binary-solvent solubility`.
4. می‌توانید ابتدا **Private** انتخاب کنید تا پیش از عمومی‌سازی پرونده‌ها را بررسی کنید. گزینه افزودن README، gitignore یا license را در این مرحله انتخاب نکنید؛ این فایل‌ها از قبل در بسته حاضرند.
5. گزینه **Create repository** را بزنید. بسته ZIP حاضر را روی دستگاه خود **Extract** کنید. فقط **محتوای داخل پوشه** را بارگذاری کنید، نه خود ZIP یا پوشه پدر را.
6. در GitHub گزینه **uploading an existing file** یا **Add file → Upload files** را انتخاب کنید، همه فایل‌ها و پوشه `results` را بکشید و در انتها **Commit changes** را بزنید. اگر فایل مخفی `.gitignore` منتقل نشد، آن را جداگانه بسازید.
7. در صفحه مخزن تأیید کنید `README.md`، `LICENSE`، `CITATION.cff`، `requirements.txt`، فایل‌های `.py` و پوشه `results/` در سطح ریشه دیده می‌شوند.
8. اطمینان بگیرید فایل‌های `MixtureSolDB.csv`، `harmonized_metadata.csv`، `private_*` یا ZIP آرشیو خصوصی در مخزن نیستند.
9. از **Settings → General → Danger Zone → Change repository visibility** مخزن را در صورت Private بودن به **Public** تبدیل کنید.

## B — اتصال Zenodo
10. در https://zenodo.org حساب بسازید یا وارد شوید؛ ترجیحاً ORCID خود را نیز در Linked accounts متصل کنید.
11. از منوی پروفایل Zenodo به **Linked accounts** بروید و GitHub را **Connect** کنید.
12. از منوی پروفایل Zenodo گزینه **GitHub** را انتخاب کنید، **Sync now** را بزنید، مخزن CHEM-02 را پیدا کرده و سوییچ آن را **Enable** کنید.
13. قبل از انتشار Release از فعال بودن سوییچ مطمئن شوید.

## C — اولین Release و DOI
14. در مخزن GitHub گزینه **Releases → Draft a new release** را بزنید.
15. در **Choose a tag** مقدار `v1.0.0` را ایجاد کنید؛ target باید `main` باشد.
16. Title: `CHEM-02 numerical benchmark — reproducibility code v1.0.0`.
17. متن `RELEASE_NOTES.md` را در Description وارد کنید. آن را به‌عنوان pre-release علامت نزنید، مگر آن‌که عمداً نسخه آزمایشی منتشر می‌کنید.
18. **Publish release** را بزنید. صبر کنید تا Zenodo Release را پردازش کند.
19. در Zenodo قسمت GitHub یا Search، رکورد نرم‌افزار را پیدا کنید و DOI **نسخه** را کپی کنید؛ آن را با DOI **دیتاست** اشتباه نگیرید.
20. DOI واقعی و URL مخزن را در مقاله و Supplementary درج و پیش از ارسال Elsevier فایل‌ها را مجدداً کنترل کنید.

**مهم:** فایل ZIP کد را به‌عنوان یک مخزن *به‌تنهایی* در صفحه GitHub آپلود نکنید؛ ساختار فایل‌های بازشده باید در ریشه مخزن قابل مشاهده باشد.

**قید علمی:** مطالعه روی داده عمومی انجام شده و ارزیابی retrospective است؛ همه آموزش‌های پرهزینه ML در دستگاهی کاملاً مستقل دوباره اجرا نشده‌اند. این واقعیت را پنهان نکنید.

راهنماهای رسمی:
- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- https://help.zenodo.org/docs/profile/linking-accounts/
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
