# Image Normalization for Hebrew Handwriting Dataset

## סקירה כללית

סקריפט זה מבצע נורמליזציה של תמונות כתב יד עברי לצורך הכנת dataset לפרויקט Computer Vision.

## התהליכים שמבוצעים

1. **Binarization (בינריזציה)** - המרת התמונה לשחור-לבן מוחלט
   - שיטות זמינות: `otsu`, `adaptive`, `sauvola`
   - ברירת מחדל: `adaptive` (מתאים לתמונות עם תאורה משתנה)

2. **Noise Removal (הסרת רעש)** - הסרת נקודות קטנות ואפרוריות
   - משתמש ב-morphological operations
   - מנתח connected components
   - מסיר רכיבים קטנים מתחת לסף (ברירת מחדל: 10 פיקסלים)

3. **Content-based Cropping (חיתוך)** - הסרת שוליים לבנים מיותרים
   - מזהה את התוכן האמיתי בתמונה
   - משאיר padding של 20 פיקסלים מסביב לתוכן

## שימוש בסיסי

```bash
# עיבוד כל התמונות בתיקייה
python normalize_images.py input_directory output_directory
```

## אופציות מתקדמות

```bash
# בחירת שיטת בינריזציה
python normalize_images.py input_dir output_dir --binarization-method otsu

# ללא הסרת רעש
python normalize_images.py input_dir output_dir --no-denoise

# ללא חיתוך
python normalize_images.py input_dir output_dir --no-crop

# נורמליזציה של גובה (שימושי למודלי Deep Learning)
python normalize_images.py input_dir output_dir --normalize-height --target-height 64

# שינוי רגישות הסרת רעש (רכיבים קטנים יותר)
python normalize_images.py input_dir output_dir --min-noise-size 5
```

## דרישות

```bash
pip install opencv-python numpy
```

## תוצאות

הסקריפט עיבד **5,572 תמונות** בהצלחה מלאה (0 כשלונות).

כל התמונות המעובדות נמצאות בתיקייה: `normalized_output/`

## דוגמאות לפני ואחרי

התמונות המקוריות היו עם:
- רקע אפור לא אחיד
- רעש ונקודות קטנות
- שוליים לבנים גדולים

התמונות המעובדות:
- רקע לבן נקי לחלוטין
- טקסט שחור חד
- חיתוך צמוד לתוכן

## מידע טכני

### פרמטרים ניתנים לשינוי בקוד

בקובץ `normalize_images.py`, ניתן לשנות:
- `padding` בפונקציה `crop_to_content()` (ברירת מחדל: 20)
- `kernel` size בפונקציה `remove_noise()` (ברירת מחדל: 2x2)
- פרמטרי ה-threshold ב-`binarize_image()`

### אלגוריתמים בשימוש

1. **Adaptive Thresholding** - עבור בינריזציה מקומית
2. **Morphological Opening** - להסרת רעש קטן
3. **Connected Components Analysis** - לזיהוי והסרת אובייקטים קטנים
4. **Bounding Rectangle** - לחישוב גבולות התוכן

## המשך העבודה

השלבים הבאים בפרויקט:
1. ✅ נורמליזציה של תמונות
2. ⏳ תיוג features גרפולוגיים (Slant, Thickness, Baseline, Word/Letter Spacing)
3. ⏳ בניית מודל ML/DL לניתוח כתב יד

## פתרון בעיות

### בעיה: התמונה יוצאת שחורה לגמרי
- **פתרון**: נסה שיטת בינריזציה אחרת (`--binarization-method otsu`)

### בעיה: חלק מהטקסט נעלם
- **פתרון**: הקטן את `--min-noise-size` (נסה 5 במקום 10)

### בעיה: יש עדיין הרבה רעש
- **פתרון**: הגדל את `--min-noise-size` (נסה 15 או 20)

## רישיון וקרדיטים

פותח עבור פרויקט Hebrew Handwriting Analysis
שנת לימודים ג', מדעי המחשב
