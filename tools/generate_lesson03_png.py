from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parents[1] / "assets/images/gemini-videos-01-03"
FA_DIR = ROOT / "png-fa"
EN_DIR = ROOT / "png-en"
FA_DIR.mkdir(parents=True, exist_ok=True)
EN_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1600, 900
BG = (248, 251, 254)
NAVY = (19, 62, 112)
BLUE = (39, 126, 207)
LIGHT = (235, 245, 253)
GRAY = (91, 106, 124)
BORDER = (174, 203, 232)
BLACK = (30, 34, 40)
WHITE = (255, 255, 255)
RED = (205, 72, 72)
GREEN = (53, 145, 90)


def pick(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

FA = pick("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansArabicUI-Regular.ttf")
FA_B = pick("/usr/share/fonts/truetype/noto/NotoKufiArabic-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf")
EN = pick("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
EN_B = pick("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
MATH = EN


def F(path, size):
    return ImageFont.truetype(path, size)


def text(d, xy, s, font, fill=BLACK, anchor="la", rtl=False):
    kw = {"font": font, "fill": fill, "anchor": anchor}
    if rtl:
        kw.update(direction="rtl", language="fa")
    d.text(xy, s, **kw)


def box(d, coords, fill=WHITE, outline=BORDER, width=3, radius=20):
    d.rounded_rectangle(coords, radius=radius, fill=fill, outline=outline, width=width)


def circle(d, center, r=25, fill=WHITE, outline=NAVY, width=3):
    x, y = center
    d.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline, width=width)


def arrow(d, a, b, fill=NAVY, width=6, head=18):
    d.line([a, b], fill=fill, width=width)
    ang = math.atan2(b[1]-a[1], b[0]-a[0])
    p1 = (b[0]-head*math.cos(ang)+head*.55*math.sin(ang), b[1]-head*math.sin(ang)-head*.55*math.cos(ang))
    p2 = (b[0]-head*math.cos(ang)-head*.55*math.sin(ang), b[1]-head*math.sin(ang)+head*.55*math.cos(ang))
    d.polygon([b, p1, p2], fill=fill)


def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def heading(d, lang, main, sub):
    if lang == "fa":
        text(d, (1530, 48), main, F(FA_B, 52), NAVY, "ra", True)
        text(d, (1530, 118), sub, F(FA, 30), GRAY, "ra", True)
    else:
        text(d, (65, 48), main, F(EN_B, 50), NAVY)
        text(d, (65, 118), sub, F(EN, 29), GRAY)


def neuron(d, area, lang):
    x0, y0, x1, y1 = area
    cx, cy = x0+505, y0+300
    for i, (yy, lab, wlab) in enumerate([(y0+175, "x₁", "w₁"), (y0+300, "x₂", "w₂"), (y0+425, "xₙ", "wₙ")]):
        circle(d, (x0+120, yy), 27)
        text(d, (x0+120, yy), lab, F(MATH, 30), anchor="mm")
        target = (cx-92, cy+(-65 if i == 0 else 0 if i == 1 else 65))
        arrow(d, (x0+150, yy), target, width=5)
        text(d, ((x0+150+target[0])/2, (yy+target[1])/2-18), wlab, F(MATH, 24), anchor="mm")
    circle(d, (cx, y0+108), 24)
    text(d, (cx, y0+108), "b", F(MATH, 27), anchor="mm")
    text(d, (cx, y0+65), "bias", F(EN, 21), anchor="mm")
    arrow(d, (cx, y0+135), (cx, cy-88), width=5)
    circle(d, (cx, cy), 94, fill=LIGHT)
    text(d, (cx, cy), "Σ", F(MATH, 58), anchor="mm")
    label = "Weighted sum + bias" if lang == "en" else "جمع وزن‌دار + bias"
    text(d, (cx, cy+122), label, F(EN, 22) if lang == "en" else F(FA, 23), anchor="mm", rtl=(lang == "fa"))
    ax0, ay0 = cx+125, cy-55
    box(d, (ax0, ay0, ax0+125, ay0+110), outline=NAVY, width=4, radius=16)
    text(d, (ax0+62, cy), "f(·)", F(MATH, 37), anchor="mm")
    text(d, (ax0+62, ay0-35), "Activation function" if lang == "en" else "تابع فعال‌سازی", F(EN, 20) if lang == "en" else F(FA, 21), anchor="mm", rtl=(lang == "fa"))
    arrow(d, (cx+96, cy), (ax0, cy), width=5)
    arrow(d, (ax0+125, cy), (ax0+215, cy), width=5)
    text(d, (ax0+235, cy), "y", F(MATH, 31), anchor="mm")
    text(d, ((x0+x1)/2, y1-62), "y = f(b + Σᵢ wᵢxᵢ)", F(MATH, 36), anchor="mm")


def fig1(lang, out):
    im, d = canvas()
    heading(d, lang,
            "یک نورون منفرد چه چیزی را محاسبه می‌کند؟" if lang == "fa" else "What Does a Single Neuron Compute?",
            "جمع وزن‌دار + bias، سپس تابع فعال‌سازی" if lang == "fa" else "Weighted sum + bias, then an activation function")
    if lang == "fa":
        left, right = (65, 180, 1015, 785), (1060, 180, 1535, 785)
        box(d, left); box(d, right)
        text(d, (980, 215), "زوم روی یک نورون", F(FA_B, 33), NAVY, "ra", True)
        text(d, (1500, 215), "یک شبکه عصبی", F(FA_B, 33), NAVY, "ra", True)
        neuron(d, left, lang)
        pts = [(1135, 340), (1135, 500), (1285, 305), (1285, 440), (1285, 590), (1450, 365), (1450, 535)]
        for a in pts[:2]:
            for b in pts[2:5]: d.line([a,b], fill=(170,185,200), width=2)
        for a in pts[2:5]:
            for b in pts[5:]: d.line([a,b], fill=(170,185,200), width=2)
        for p in pts: circle(d, p, 23)
        circle(d, (1285, 440), 28, fill=(111,185,245))
        text(d, (1285, 700), "زوم روی یک نورون", F(FA, 25), anchor="mm", rtl=True)
        arrow(d, (1045, 490), (1020, 490), width=5)
    else:
        left, right = (65, 180, 535, 785), (585, 180, 1535, 785)
        box(d, left); box(d, right)
        text(d, (95, 215), "A neural network", F(EN_B, 33), NAVY)
        text(d, (620, 215), "Zoom in on one neuron", F(EN_B, 33), NAVY)
        pts = [(145,340),(145,500),(300,305),(300,440),(300,590),(455,365),(455,535)]
        for a in pts[:2]:
            for b in pts[2:5]: d.line([a,b], fill=(170,185,200), width=2)
        for a in pts[2:5]:
            for b in pts[5:]: d.line([a,b], fill=(170,185,200), width=2)
        for p in pts: circle(d,p,23)
        circle(d,(300,440),28,fill=(111,185,245))
        text(d,(300,700),"Zoom in on one neuron",F(EN,23),anchor="mm")
        arrow(d,(550,490),(580,490),width=5)
        neuron(d,right,lang)
    im.save(out, optimize=True)


def fig2(lang, out):
    im,d=canvas()
    heading(d,lang,"یک نورون برای Regression" if lang=="fa" else "A Single Neuron for Regression","فعال‌سازی خطی یک خط مستقیم می‌سازد" if lang=="fa" else "A linear activation produces a straight line")
    tb=(1040,190,1535,790) if lang=="fa" else (65,190,555,790)
    gb=(65,190,990,790) if lang=="fa" else (610,190,1535,790)
    box(d,tb); box(d,gb)
    if lang=="fa": text(d,(1500,225),"مثال خطی",F(FA_B,31),NAVY,"ra",True)
    else: text(d,(100,225),"Linear example",F(EN_B,31),NAVY)
    lines=["x = x₁w₁ + b","y = x","","w₁ = 3","b = -2","","y = 3x₁ - 2"]
    yy=290
    for s in lines:
        text(d,(1500,yy) if lang=="fa" else (100,yy),s,F(MATH,31),anchor="ra" if lang=="fa" else "la")
        yy+=66
    x0,y0,x1,y1=gb; ox=x0+125; oy=y1-110
    d.line((ox,y0+80,ox,oy),fill=BLACK,width=3); d.line((ox,oy,x1-75,oy),fill=BLACK,width=3)
    arrow(d,(x1-120,oy),(x1-75,oy),BLACK,3,14); arrow(d,(ox,y0+125),(ox,y0+80),BLACK,3,14)
    text(d,(x1-55,oy+10),"x₁",F(MATH,25),anchor="mm"); text(d,(ox-20,y0+66),"y",F(MATH,25),anchor="mm")
    def mp(x,y): return ox+(x+.5)/3*(x1-ox-145), oy-(y+4)/10*(oy-y0-120)
    pts=[mp(-.5+3*i/250,3*(-.5+3*i/250)-2) for i in range(251)]
    d.line(pts,fill=BLUE,width=6)
    p0,p1=mp(0,-2),mp(1,1)
    for p in [p0,p1]: d.ellipse((p[0]-8,p[1]-8,p[0]+8,p[1]+8),fill=RED)
    text(d,(p0[0]+25,p0[1]+14),"-2",F(MATH,22))
    text(d,(p1[0]+28,p1[1]-30),"شیب = 3" if lang=="fa" else "slope = 3",F(FA,22) if lang=="fa" else F(EN,22),rtl=(lang=="fa"))
    im.save(out,optimize=True)


def fig3(lang,out):
    im,d=canvas()
    heading(d,lang,"همان نورون برای Classification" if lang=="fa" else "The Same Neuron for Classification","فقط تابع فعال‌سازی را به تابع پله‌ای تغییر می‌دهیم" if lang=="fa" else "Keep the weighted sum; change the activation to a step function")
    box(d,(65,190,1535,790))
    for yy,lab,wlab,target_y in [(355,"x₁","w₁",425),(540,"x₂","w₂",475)]:
        circle(d,(210,yy),30); text(d,(210,yy),lab,F(MATH,30),anchor="mm")
        arrow(d,(244,yy),(605,target_y),width=6)
        text(d,(410,(yy+target_y)/2-18),wlab,F(MATH,25),anchor="mm")
    circle(d,(700,450),105,fill=LIGHT); text(d,(700,450),"Σ",F(MATH,65),anchor="mm")
    text(d,(700,585),"جمع وزن‌دار" if lang=="fa" else "Weighted sum",F(FA,27) if lang=="fa" else F(EN,27),anchor="mm",rtl=(lang=="fa"))
    box(d,(970,370,1145,535),outline=NAVY,width=4,radius=18)
    d.line((1010,485,1060,485,1060,415,1110,415),fill=BLUE,width=8)
    text(d,(1058,330),"تابع پله‌ای" if lang=="fa" else "Step function",F(FA_B,26) if lang=="fa" else F(EN_B,26),NAVY,anchor="mm",rtl=(lang=="fa"))
    arrow(d,(808,450),(965,450),width=6); arrow(d,(1150,450),(1370,450),width=6)
    text(d,(1410,450),"y",F(MATH,34),anchor="mm")
    text(d,(520,690),"y = 0   اگر x ≤ 0" if lang=="fa" else "y = 0   if x ≤ 0",F(MATH,27),anchor="mm")
    text(d,(1035,690),"y = 1   اگر x > 0" if lang=="fa" else "y = 1   if x > 0",F(MATH,27),anchor="mm")
    im.save(out,optimize=True)


def fig4(lang,out):
    im,d=canvas()
    heading(d,lang,"مرز تصمیم در مثال دو ورودی" if lang=="fa" else "Decision Boundary in the Two-Input Example","w₁ = 2 ، w₂ = -1 ، b = 1" if lang=="fa" else "w₁ = 2, w₂ = -1, b = 1")
    tb=(1050,190,1535,790) if lang=="fa" else (65,190,555,790)
    gb=(65,190,995,790) if lang=="fa" else (610,190,1535,790)
    box(d,tb); box(d,gb)
    if lang=="fa": text(d,(1500,225),"پارامترها و مرز",F(FA_B,29),NAVY,"ra",True)
    else: text(d,(100,225),"Parameters and boundary",F(EN_B,28),NAVY)
    lines=["w₁ = 2","w₂ = -1","b = 1","","x = 2x₁ - x₂ + 1","","2x₁ - x₂ + 1 = 0","x₂ = 2x₁ + 1"]
    yy=285
    for s in lines:
        text(d,(1500,yy) if lang=="fa" else (100,yy),s,F(MATH,28),anchor="ra" if lang=="fa" else "la"); yy+=58
    x0,y0,x1,y1=gb; ox=x0+115; oy=y1-105
    d.line((ox,y0+85,ox,oy),fill=BLACK,width=3); d.line((ox,oy,x1-70,oy),fill=BLACK,width=3)
    arrow(d,(x1-115,oy),(x1-70,oy),BLACK,3,14); arrow(d,(ox,y0+130),(ox,y0+85),BLACK,3,14)
    text(d,(x1-50,oy+10),"x₁",F(MATH,24),anchor="mm"); text(d,(ox-20,y0+70),"x₂",F(MATH,24),anchor="mm")
    def mp(x,y): return ox+(x+2)/4*(x1-ox-135), oy-(y+2)/6*(oy-y0-130)
    pts=[mp(-2+4*i/250,2*(-2+4*i/250)+1) for i in range(251)]
    d.line(pts,fill=RED,width=6)
    for x,y in [(-1.6,1.2),(-1.1,2.1),(-.5,2.8)]:
        px,py=mp(x,y); d.ellipse((px-8,py-8,px+8,py+8),outline=NAVY,width=3,fill=WHITE)
    for x,y in [(-.6,-.5),(.1,.4),(.7,.9),(1.2,.2),(1.5,1.4)]:
        px,py=mp(x,y); d.line((px-9,py-9,px+9,py+9),fill=GREEN,width=4); d.line((px-9,py+9,px+9,py-9),fill=GREEN,width=4)
    text(d,(x0+160,y0+105),"y = 0",F(MATH,25)); text(d,(x1-210,y1-155),"y = 1",F(MATH,25))
    im.save(out,optimize=True)


def fig5(lang,out):
    im,d=canvas()
    heading(d,lang,"مرز تصمیم در یک بعد" if lang=="fa" else "Decision Boundary in One Dimension","مرز تصمیم حالا یک نقطه روی خط اعداد است" if lang=="fa" else "The decision boundary is now a point on the number line")
    box(d,(65,190,1535,790))
    pos=(1450,260) if lang=="fa" else (115,260); anchor="ra" if lang=="fa" else "la"
    for i,s in enumerate(["x = 2x₁ + 1","2x₁ + 1 = 0","x₁ = -0.5"]):
        text(d,(pos[0],pos[1]+i*70),s,F(MATH,31),RED if i==2 else BLACK,anchor=anchor)
    x0,x1,y=210,1390,590
    d.line((x0,y,x1,y),fill=BLACK,width=4); arrow(d,(x1-42,y),(x1,y),BLACK,4,15)
    def m(v): return x0+(v+2)/4*(x1-x0-45)
    for v in [-2,-1,-.5,0,1,2]:
        xx=m(v); d.line((xx,y-15,xx,y+15),fill=BLACK,width=3); text(d,(xx,y+32),str(v),F(EN,20),anchor="ma")
    bx=m(-.5); d.line((bx,y-78,bx,y+78),fill=RED,width=6)
    text(d,(bx,y-102),"مرز" if lang=="fa" else "boundary",F(FA,22) if lang=="fa" else F(EN,22),RED,anchor="mm",rtl=(lang=="fa"))
    text(d,(m(-1.25),y-55),"y = 0",F(MATH,25),anchor="mm"); text(d,(m(.9),y-55),"y = 1",F(MATH,25),anchor="mm")
    im.save(out,optimize=True)


def fig6(lang,out):
    im,d=canvas()
    heading(d,lang,"از داده تا تابع زیان" if lang=="fa" else "From Data to the Loss Function","Loss نشان می‌دهد مدل با پارامترهای فعلی چقدر اشتباه می‌کند" if lang=="fa" else "Loss measures how wrong the current model is")
    labels = ["مجموعه‌داده\n(x, y)","مدل\ny_hat = Model(x)","خطا\ny - Model(x)","توان دو + جمع"] if lang=="fa" else ["Dataset\n(x, y)","Model\ny_hat = Model(x)","Error\ny - Model(x)","Square + sum"]
    xs=[80,440,800,1160]
    for i,x in enumerate(xs):
        box(d,(x,250,x+290,430))
        p=labels[i].split("\n")
        if lang=="fa": text(d,(x+145,305),p[0],F(FA,27),anchor="mm",rtl=True)
        else: text(d,(x+145,305),p[0],F(EN,27),anchor="mm")
        if len(p)>1: text(d,(x+145,365),p[1],F(MATH,24),anchor="mm")
        if i<3: arrow(d,(x+290,340),(xs[i+1]-18,340),width=5)
    box(d,(145,535,1455,800),fill=LIGHT)
    text(d,(800,590),"تابع زیان" if lang=="fa" else "Loss function",F(FA_B,31) if lang=="fa" else F(EN_B,31),NAVY,anchor="mm",rtl=(lang=="fa"))
    text(d,(800,660),"L(w₁, …, wₙ, b) = Σ₍(x,y)∈D₎ (y − Model(x))²",F(MATH,31),anchor="mm")
    note="هدف آموزش: وزن‌ها و bias را طوری تغییر دهیم که Loss کوچک شود." if lang=="fa" else "Training changes the weights and bias so that the loss becomes smaller."
    text(d,(800,740),note,F(FA,25) if lang=="fa" else F(EN,25),anchor="mm",rtl=(lang=="fa"))
    im.save(out,optimize=True)


for lang, directory in [("fa", FA_DIR), ("en", EN_DIR)]:
    fig1(lang, directory/"lesson-03-01.png")
    fig2(lang, directory/"lesson-03-02.png")
    fig3(lang, directory/"lesson-03-03.png")
    fig4(lang, directory/"lesson-03-04.png")
    fig5(lang, directory/"lesson-03-05.png")
    fig6(lang, directory/"lesson-03-06.png")

print("generated", FA_DIR, EN_DIR)
