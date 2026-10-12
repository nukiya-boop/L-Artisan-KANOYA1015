#!/usr/bin/env python3
"""L'Artisan KANOYA — ある日のディナー「葛切り、春菊、あさりだし」
Instagram リール用 PR 動画 (1080x1920 / 30fps / 30秒) を生成するスクリプト。

使い方:
    python3 video/make_reel.py --fonts <フォントのディレクトリ> [--out video/kuzukiri_reel.mp4]

必要なフォント (Google Fonts / OFL):
    ShipporiMincho-Medium.ttf, ShipporiMincho-Regular.ttf,
    CormorantGaramond-Italic[wght].ttf -> CormorantItalic.ttf
"""
import argparse
import glob
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H = 1080, 1920
FPS = 30
DUR = 30.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = (214, 186, 130)
IVORY = (246, 240, 228)


# ---------------------------------------------------------------- utilities
def ease(t):  # easeInOutCubic
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def load(num, long_side=3000, rotate=False):
    path = glob.glob(os.path.join(ROOT, "images", f"*{num}.jpg"))[0]
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if rotate:  # 真上からの写真は 90° 回して縦位置に (切れる範囲を減らす)
        im = im.transpose(Image.ROTATE_270)
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    return im


def grade(im):
    """しっとりした色調: 黒を少し持ち上げ、ハイライトを暖かく。"""
    a = np.asarray(im).astype(np.float32) / 255.0
    a = 0.03 + a * 0.97
    a[..., 0] *= 1.02
    a[..., 2] *= 0.98
    a = np.clip(a, 0, 1) ** 0.98
    return Image.fromarray((a * 255).astype(np.uint8))


def view(src, cx, cy, zoom, rot=0.0, out=(W, H)):
    """src 上の (cx,cy)[0-1] を中心に、出力いっぱいに cover する視点で切り出す。
    zoom=1 で短辺がちょうど収まる大きさ。"""
    sw, sh = src.size
    ow, oh = out
    base = max(ow / sw, oh / sh)
    s = base * zoom  # 出力px / 元px
    # 出力座標 -> 元座標 の逆変換 (回転付き)
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    px, py = cx * sw, cy * sh
    # 中心を画面外に出さない
    halfw, halfh = ow / 2 / s, oh / 2 / s
    px = min(max(px, halfw), sw - halfw)
    py = min(max(py, halfh), sh - halfh)
    a, b = c / s, si / s
    d, e = -si / s, c / s
    tx = px - a * ow / 2 - b * oh / 2
    ty = py - d * ow / 2 - e * oh / 2
    return src.transform(out, Image.AFFINE, (a, b, tx, d, e, ty), resample=Image.BILINEAR)


def to_arr(im):
    return np.asarray(im).astype(np.float32) / 255.0


# ---------------------------------------------------------------- text
class Fonts:
    def __init__(self, d):
        self.dir = d

    def mincho(self, size, medium=True):
        name = "ShipporiMincho-Medium.ttf" if medium else "ShipporiMincho-Regular.ttf"
        return ImageFont.truetype(os.path.join(self.dir, name), size)

    def italic(self, size, weight=400):
        f = ImageFont.truetype(os.path.join(self.dir, "CormorantItalic.ttf"), size)
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
        return f


def text_layer(items, shadow=True):
    """items: [(text, font, (x,y), anchor, color, tracking)] を RGBA レイヤーに描画。"""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    for text, font, (x, y), anchor, color, tracking in items:
        if tracking == 0:
            dr.text((x, y), text, font=font, fill=color + (255,), anchor=anchor)
            continue
        widths = [dr.textlength(ch, font=font) for ch in text]
        total = sum(widths) + tracking * (len(text) - 1)
        sx = x - total / 2 if anchor[0] == "m" else (x - total if anchor[0] == "r" else x)
        for ch, w in zip(text, widths):
            dr.text((sx, y), ch, font=font, fill=color + (255,), anchor="l" + anchor[1])
            sx += w + tracking
    return add_shadow(layer) if shadow else layer


def vertical_chars(text, font, x, y, gap, color):
    """縦書き: 1文字ずつのレイヤー (文字ごとに時間差で出すため) を返す。"""
    out = []
    for i, ch in enumerate(text):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        cy = y + i * gap
        if ch in "、。":
            dr.text((x + gap * 0.28, cy - gap * 0.28), ch, font=font, fill=color + (255,), anchor="mm")
        elif ch == "ー":
            l2 = Image.new("RGBA", (gap, gap), (0, 0, 0, 0))
            ImageDraw.Draw(l2).text((gap / 2, gap / 2), ch, font=font, fill=color + (255,), anchor="mm")
            l2 = l2.rotate(-90)
            layer.alpha_composite(l2, (int(x - gap / 2), int(cy - gap / 2)))
        else:
            dr.text((x, cy), ch, font=font, fill=color + (255,), anchor="mm")
        out.append(add_shadow(layer))
    return out


def add_shadow(layer, radius=14, strength=0.55):
    a = layer.split()[-1]
    sh = a.filter(ImageFilter.GaussianBlur(radius))
    sh = sh.point(lambda v: int(v * strength))
    base = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    base.putalpha(sh)
    base.alpha_composite(layer)
    return base


def line_layer(x0, y0, x1, y1, color=GOLD, width=2):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).line([(x0, y0), (x1, y1)], fill=color + (255,), width=width)
    return layer


class Layer:
    """RGBA レイヤーを numpy で保持し、透明度・縦移動で合成する。"""

    def __init__(self, im):
        a = np.asarray(im).astype(np.float32) / 255.0
        self.rgb = a[..., :3]
        self.a = a[..., 3:4]

    def over(self, frame, alpha, dy=0):
        if alpha <= 0.002:
            return frame
        rgb, a = self.rgb, self.a
        if dy:
            dy = int(round(dy))
            rgb = np.roll(rgb, dy, axis=0)
            a = np.roll(a, dy, axis=0)
        k = a * alpha
        return frame * (1 - k) + rgb * k


def fade_in_out(t, t0, t1, fi=0.6, fo=0.5):
    return clamp01((t - t0) / fi) * clamp01((t1 - t) / fo)


# ---------------------------------------------------------------- scenes
class Scene:
    def __init__(self, start, end):
        self.start, self.end = start, end

    def p(self, t):
        return clamp01((t - self.start) / (self.end - self.start))


class FullBleed(Scene):
    """全面表示 + ゆっくりとしたカメラワーク。"""

    def __init__(self, start, end, im, c0, c1, z0, z1, r0=0.0, r1=0.0, focus_pull=False, blur_from=None):
        super().__init__(start, end)
        self.im, self.c0, self.c1, self.z0, self.z1, self.r0, self.r1 = im, c0, c1, z0, z1, r0, r1
        self.focus_pull = focus_pull
        self.blur_from = blur_from

    def render(self, t):
        p = self.p(t)
        q = 0.5 - 0.5 * math.cos(math.pi * p)  # なめらかに等速寄り
        cx = self.c0[0] + (self.c1[0] - self.c0[0]) * q
        cy = self.c0[1] + (self.c1[1] - self.c0[1]) * q
        z = self.z0 + (self.z1 - self.z0) * q
        r = self.r0 + (self.r1 - self.r0) * q
        fr = view(self.im, cx, cy, z, r)
        if self.focus_pull:
            b = 18 * (1 - ease_out((t - self.start) / 2.4))
            if b > 0.3:
                fr = fr.filter(ImageFilter.GaussianBlur(b))
        if self.blur_from is not None:
            b = 16 * ease((t - self.blur_from) / 1.6)
            if b > 0.3:
                fr = fr.filter(ImageFilter.GaussianBlur(b))
        return to_arr(fr)


class Framed(Scene):
    """写真を切らずに全体を見せる「額装」レイアウト。背景は同じ写真をぼかしたもの。"""

    def __init__(self, start, end, im, frame_y=0.5, width=940, rise=60):
        super().__init__(start, end)
        self.im = im
        bg = view(im, 0.5, 0.5, 1.0).filter(ImageFilter.GaussianBlur(40))
        self.bg = to_arr(bg) * 0.42
        fw = width
        fh = int(im.size[1] * fw / im.size[0])
        self.fw, self.fh, self.frame_y, self.rise = fw, fh, frame_y, rise
        self.big = im.resize((int(fw * 1.12), int(fh * 1.12)), Image.LANCZOS)

    def render(self, t):
        p = self.p(t)
        fr = self.bg.copy()
        # 写真内をゆっくりズームアウト (1.10 -> 1.0)
        z = 1.10 - 0.10 * ease_out(p * 1.2)
        ph = view(self.big, 0.5, 0.5, z, out=(self.fw, self.fh))
        x0 = (W - self.fw) // 2
        y0 = int(H * self.frame_y - self.fh / 2 + self.rise * (1 - ease_out(p * 1.6)))
        # 影
        shadow = np.zeros((H, W), np.float32)
        shadow[y0 + 24:y0 + self.fh + 24, x0 + 10:x0 + self.fw + 10] = 1
        shadow = np.asarray(Image.fromarray((shadow * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(28))).astype(np.float32)[..., None] / 255.0
        fr = fr * (1 - 0.6 * shadow)
        fr[y0:y0 + self.fh, x0:x0 + self.fw] = to_arr(ph)
        # 細い金の罫 (額縁の内側に少し離して)
        m = 16
        g = np.array(GOLD, np.float32) / 255.0
        a = 0.55
        for (ys, ye, xs, xe) in [(y0 - m, y0 - m + 1, x0 - m, x0 + self.fw + m),
                                 (y0 + self.fh + m, y0 + self.fh + m + 1, x0 - m, x0 + self.fw + m),
                                 (y0 - m, y0 + self.fh + m + 1, x0 - m, x0 - m + 1),
                                 (y0 - m, y0 + self.fh + m + 1, x0 + self.fw + m, x0 + self.fw + m + 1)]:
            fr[ys:ye, xs:xe] = fr[ys:ye, xs:xe] * (1 - a) + g * a
        return fr


# ---------------------------------------------------------------- transitions
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def tr_iris(a, b, p, cx=0.5, cy=0.6, soft=90):
    """円形に広がる (お椀の形に合わせて)。"""
    r = ease(p) * math.hypot(W, H) * 0.75
    d = np.sqrt((XX - cx * W) ** 2 + (YY - cy * H) ** 2)
    m = np.clip((r - d) / soft + 0.5, 0, 1)[..., None]
    return a * (1 - m) + b * m


def tr_strips(a, b, p, n=5):
    """縦の短冊が時間差で上から降りてくる。"""
    out = a.copy()
    sw = W / n
    for i in range(n):
        pi = ease(clamp01(p * 1.6 - i * 0.15))
        if pi <= 0:
            continue
        x0, x1 = int(i * sw), int((i + 1) * sw)
        hh = int(pi * H)
        if hh <= 0:
            continue
        out[:hh, x0:x1] = b[:hh, x0:x1]
        # 境目に金の細線
        y = min(hh, H - 2)
        if pi < 0.999:
            out[y:y + 2, x0:x1] = out[y:y + 2, x0:x1] * 0.3 + np.array(GOLD, np.float32) / 255 * 0.7
    return out


def tr_blur_dissolve(a, b, p):
    """ピントが外れながら溶けて、次の写真で合う。"""
    e = ease(p)
    rad = 22 * math.sin(math.pi * e)

    def blur(x):
        if rad < 0.5:
            return x
        im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
        return to_arr(im.filter(ImageFilter.GaussianBlur(rad)))

    return blur(a) * (1 - e) + blur(b) * e


def tr_diagonal(a, b, p, soft=260):
    """斜めの柔らかなワイプ。"""
    e = ease(p)
    proj = (XX / W * 0.55 + (H - YY) / H * 0.45)  # 0..1 (左下 -> 右上)
    edge = -0.3 + e * 1.6
    m = np.clip((edge - proj) * W / soft + 0.5, 0, 1)[..., None]
    return a * (1 - m) + b * m


def tr_light_leak(a, b, p):
    """暖色の光が差し込むクロスフェード。"""
    e = ease(p)
    glow = math.sin(math.pi * e)
    cx = W * (1.1 - 0.9 * e)
    leak = np.exp(-(((XX - cx) / (W * 0.55)) ** 2 + ((YY - H * 0.35) / (H * 0.6)) ** 2))[..., None]
    col = np.array([1.0, 0.72, 0.42], np.float32)
    mix = a * (1 - e) + b * e
    return np.clip(mix + leak * col * glow * 0.28, 0, 1)


def tr_open_lid(a, b, p):
    """前の画面が上下に割れて開き、次の画面が現れる (蓋を開けるイメージ)。"""
    e = ease(p)
    out = b * (0.75 + 0.25 * e)
    off = int(e * H / 2)
    half = H // 2
    if off < half:
        out[:half - off] = a[off:half]
        out[half + off:] = a[half:H - off]
        for y in (half - off, half + off):
            y = min(max(y, 1), H - 2)
            out[y - 1:y + 1] = np.array(GOLD, np.float32) / 255
    return out


def tr_fade_dark(a, b, p):
    e = ease(p)
    if e < 0.5:
        return a * (1 - e * 2 * 0.85)
    return b * (0.15 + (e - 0.5) * 2 * 0.85)


# ---------------------------------------------------------------- build
def build(fonts_dir):
    F = Fonts(fonts_dir)
    print("loading images...")
    im = {n: grade(load(n, rotate=n in ("0274", "0276", "0275")))
          for n in ["0280", "0274", "0272", "0276", "0278", "0275", "0281", "0282"]}

    scenes = [
        # 1. オープニング: 縦位置の一皿、ピントが合っていく
        FullBleed(0.0, 4.6, im["0280"], (0.5, 0.60), (0.5, 0.58), 1.18, 1.04, focus_pull=True),
        # 2. 朱の蓋と椀を俯瞰で、横へ流す
        FullBleed(3.8, 8.4, im["0274"], (0.58, 0.38), (0.46, 0.47), 1.16, 1.04),
        # 3. 写真全体を額装で
        Framed(7.6, 12.2, im["0272"], frame_y=0.43),
        # 4. 葛切り — 真上から寄る
        FullBleed(11.4, 15.6, im["0276"], (0.52, 0.50), (0.52, 0.50), 1.00, 1.18),
        # 5. 春菊 — 額装 (横から)
        Framed(14.8, 19.0, im["0278"], frame_y=0.45),
        # 6. あさりだし — ゆっくり回転
        FullBleed(18.2, 22.4, im["0275"], (0.48, 0.42), (0.48, 0.42), 1.14, 1.06, r0=-3.0, r1=3.0),
        # 7. 料理名
        FullBleed(21.6, 26.2, im["0281"], (0.5, 0.58), (0.5, 0.62), 1.02, 1.12),
        # 8. エンドカード
        FullBleed(25.4, 30.0, im["0282"], (0.5, 0.62), (0.5, 0.62), 1.25, 1.12, blur_from=25.8),
    ]
    transitions = [tr_iris, tr_strips, tr_blur_dissolve, tr_diagonal, tr_light_leak, tr_open_lid, tr_fade_dark]
    tr_kwargs = [dict(cx=0.58, cy=0.40), {}, {}, {}, {}, {}, {}]

    # ---- テロップ
    print("rendering text...")
    T = []  # (layer, t0, t1, style)  style: 'rise' | 'fade'

    def add(items, t0, t1, style="rise", fi=0.8, fo=0.5):
        T.append((Layer(text_layer(items)), t0, t1, style, fi, fo))

    # 1: ある日のディナーより (縦書き、1文字ずつ)
    vchars = vertical_chars("ある日のディナーより", F.mincho(64), W - 150, 260, 82, IVORY)
    vlayers = [Layer(l) for l in vchars]
    add([("Un jour, un dîner", F.italic(46), (150, 300), "lm", GOLD, 2)], 1.4, 4.2, "fade", 1.0, 0.5)
    add([("L'Artisan KANOYA", F.italic(40, 500), (150, 360), "lm", IVORY, 3)], 1.8, 4.2, "fade", 1.0, 0.5)

    # 2
    add([("朱の蓋を、そっと。", F.mincho(62), (W / 2, 1560), "mm", IVORY, 10),
         ("Ouvrir doucement", F.italic(40), (W / 2, 1650), "mm", GOLD, 2)], 4.9, 8.0)
    # 3
    add([("ひと椀に、季節を纏う。", F.mincho(60), (W / 2, 1420), "mm", IVORY, 10),
         ("La saison dans un bol", F.italic(40), (W / 2, 1510), "mm", GOLD, 2)], 8.5, 11.8)
    # 4 葛切り
    add([("葛切り", F.mincho(120), (W / 2, 1568), "mm", IVORY, 30),
         ("Kuzukiri", F.italic(48), (W / 2, 1690), "mm", GOLD, 6),
         ("つるりと、透きとおる。", F.mincho(44, False), (W / 2, 1775), "mm", IVORY, 8)], 12.4, 15.1)
    # 5 春菊
    add([("春菊", F.mincho(120), (W / 2, 1404), "mm", IVORY, 30),
         ("Shungiku", F.italic(48), (W / 2, 1530), "mm", GOLD, 6),
         ("ほろ苦く、青く香る。", F.mincho(44, False), (W / 2, 1620), "mm", IVORY, 8)], 15.7, 18.5)
    # 6 あさりだし
    add([("あさりだし", F.mincho(110), (W / 2, 1572), "mm", IVORY, 22),
         ("Bouillon de palourdes", F.italic(48), (W / 2, 1690), "mm", GOLD, 4),
         ("潮の旨みを、静かに。", F.mincho(44, False), (W / 2, 1775), "mm", IVORY, 8)], 19.1, 21.9)
    # 7 料理名
    add([("— ある日のディナーより —", F.mincho(38, False), (W / 2, 250), "mm", GOLD, 6)], 22.4, 25.8, "fade")
    add([("葛切り、春菊、あさりだし", F.mincho(66), (W / 2, 340), "mm", IVORY, 8)], 22.7, 25.8, "rise", 1.0)
    # 8 エンド
    add([("L'Artisan", F.italic(120, 500), (W / 2, 760), "mm", IVORY, 4),
         ("KANOYA", F.italic(64, 500), (W / 2, 880), "mm", GOLD, 28)], 26.2, 30.5, "fade", 1.0)
    add([("四季を纏う奈良の食材と、", F.mincho(42, False), (W / 2, 1060), "mm", IVORY, 6),
         ("フランスの感性が出会う場所", F.mincho(42, False), (W / 2, 1130), "mm", IVORY, 6)], 26.9, 30.5, "rise", 1.0)
    add([("ご予約  0742-22-6666", F.mincho(40), (W / 2, 1300), "mm", IVORY, 4),
         ("lartisankanoya.com", F.italic(40), (W / 2, 1370), "mm", GOLD, 3)], 27.6, 30.5, "rise", 1.0)
    end_line = Layer(line_layer(W / 2 - 160, 975, W / 2 + 160, 975))

    # ---- 仕上げ用: ビネット, 下部のグラデ (文字の読みやすさ)
    d = np.sqrt(((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H / 2) / (H * 0.75)) ** 2)
    vign = np.clip(1 - 0.55 * d ** 2.2, 0.25, 1)[..., None]
    lowgrad = np.clip((YY - H * 0.55) / (H * 0.45), 0, 1)[..., None] ** 1.5
    topgrad = np.clip((H * 0.30 - YY) / (H * 0.30), 0, 1)[..., None] ** 1.5
    rng = np.random.default_rng(7)

    def frame_at(t):
        active = [i for i, s in enumerate(scenes) if s.start <= t < s.end + 1e-6]
        if len(active) == 1:
            fr = scenes[active[0]].render(t)
        else:
            i, j = active[0], active[-1]
            a, b = scenes[i].render(t), scenes[j].render(t)
            p = (t - scenes[j].start) / (scenes[i].end - scenes[j].start)
            fr = transitions[i](a, b, p, **tr_kwargs[i])

        fr = fr * vign
        fr = fr * (1 - 0.55 * lowgrad)
        if 0 <= t < 4.6 or 21.6 <= t < 26.2:
            fr = fr * (1 - 0.5 * topgrad)
        if t >= 25.4:  # エンドカードは全体を落として文字を主役に
            fr = fr * (1 - 0.55 * clamp01((t - 25.8) / 1.2))

        # 縦書き 1文字ずつ
        for k, L in enumerate(vlayers):
            t0 = 0.8 + k * 0.12
            al = clamp01((t - t0) / 0.6) * clamp01((4.2 - t) / 0.5)
            fr = L.over(fr, al, dy=-14 * (1 - ease_out((t - t0) / 0.8)))

        for L, t0, t1, style, fi, fo in T:
            al = fade_in_out(t, t0, t1, fi, fo)
            dy = 26 * (1 - ease_out((t - t0) / 1.2)) if style == "rise" else 0
            fr = L.over(fr, al, dy=dy)

        # 金の罫
        if 26.4 <= t:
            w = ease_out((t - 26.4) / 1.0)
            al = clamp01((t - 26.4) / 0.3)
            m = (np.abs(XX[975] - W / 2) <= 160 * w)
            row = end_line.a[975:977] * m[None, :, None]
            fr[975:977] = fr[975:977] * (1 - row * al) + end_line.rgb[975:977] * row * al
        for t0, t1, yline in [(12.6, 15.1, 1650), (15.9, 18.5, 1488), (19.3, 21.9, 1650)]:
            al = fade_in_out(t, t0, t1, 0.8, 0.5)
            if al > 0:
                w = ease_out((t - t0) / 1.0)
                m = (np.abs(XX[0] - W / 2) <= 70 * w).astype(np.float32)[None, :, None]
                g = np.array(GOLD, np.float32) / 255
                fr[yline:yline + 2] = fr[yline:yline + 2] * (1 - m * al * 0.8) + g * m * al * 0.8

        # フィルムグレイン
        noise = rng.normal(0, 0.018, (H // 2, W // 2, 1)).astype(np.float32)
        noise = noise.repeat(2, 0).repeat(2, 1)
        fr = fr + noise

        # 冒頭と最後のフェード
        fr = fr * clamp01(t / 0.6) * clamp01((DUR - t) / 0.8)
        return (np.clip(fr, 0, 1) * 255).astype(np.uint8)

    return frame_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "video", "kuzukiri_reel.mp4"))
    ap.add_argument("--music", help="BGM の音声ファイル (省略時は無音)")
    ap.add_argument("--music-start", type=float, default=9.0, help="BGM の使い始め位置 (秒)")
    ap.add_argument("--stills", help="指定すると、この秒数のフレームだけ PNG 書き出し (カンマ区切り)")
    args = ap.parse_args()

    frame_at = build(args.fonts)

    if args.stills:
        for s in args.stills.split(","):
            t = float(s)
            Image.fromarray(frame_at(t)).save(os.path.splitext(args.out)[0] + f"_{t:05.2f}.png")
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           *(["-ss", str(args.music_start), "-t", str(DUR), "-i", args.music] if args.music
             else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]),
           *(["-af", f"afade=t=in:st=0:d=1.0,afade=t=out:st={DUR - 2.8}:d=2.8,"
                     "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"] if args.music else []),
           "-c:v", "libx264", "-preset", "slow", "-b:v", "9M", "-maxrate", "12M", "-bufsize", "18M", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "192k", "-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        proc.stdin.write(frame_at(i / FPS).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("done:", args.out)


if __name__ == "__main__":
    main()
