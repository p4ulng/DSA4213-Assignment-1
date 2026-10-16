"""Save loss curves from Part I histories without a GUI or matplotlib backend."""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_history(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def font(size, bold=False):
    choices = ['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
               '/System/Library/Fonts/Supplemental/Helvetica Neue.ttc']
    for path in choices:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs/revised')
    parser.add_argument('--out', default='results/revised_loss_curves.png')
    args = parser.parse_args()
    colors = {'rnn': '#3b82f6', 'lstm': '#10b981', 'transformer': '#f97316'}
    histories = {kind: read_history(Path(args.runs) / kind / 'history.csv')
                 for kind in ('rnn', 'lstm', 'transformer')}
    points = [(kind, split, int(row['epoch']), float(row[f'{split}_loss']))
              for kind, rows in histories.items() for split in ('train', 'val') for row in rows]

    width, height = 1600, 900
    left, right, top, bottom = 115, 1540, 210, 770
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    title_font, axis_font, label_font = font(36, True), font(23), font(20)
    draw.text((width / 2, 26), 'Part I: Training and Validation Loss', fill='#111827', font=title_font, anchor='mt')
    ymin = min(value for _, _, _, value in points) - .08
    ymax = max(value for _, _, _, value in points) + .08
    ystep = .25
    ticks = [round(ymin / ystep) * ystep]
    while ticks[-1] < ymax:
        ticks.append(round(ticks[-1] + ystep, 2))
    max_epoch = max(epoch for _, _, epoch, _ in points)
    x = lambda epoch: left + (epoch - 1) / max(max_epoch - 1, 1) * (right - left)
    y = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for tick in ticks:
        yy = y(tick)
        draw.line((left, yy, right, yy), fill='#e5e7eb', width=2)
        draw.text((left - 16, yy), f'{tick:.2f}', fill='#374151', font=label_font, anchor='rm')
    for epoch in range(1, max_epoch + 1):
        xx = x(epoch)
        draw.line((xx, top, xx, bottom), fill='#f3f4f6', width=1)
        draw.text((xx, bottom + 18), str(epoch), fill='#374151', font=label_font, anchor='ma')
    draw.line((left, top, left, bottom), fill='#111827', width=3)
    draw.line((left, bottom, right, bottom), fill='#111827', width=3)
    draw.text(((left + right) / 2, 835), 'Epoch', fill='#111827', font=axis_font, anchor='ma')
    y_label = Image.new('RGBA', (440, 40), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((220, 20), 'Next-token cross-entropy (nats)', fill='#111827', font=axis_font, anchor='mm')
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (16, int((top + bottom - y_label.height) / 2)), y_label)

    for kind, rows in histories.items():
        best_epoch = min(rows, key=lambda row: float(row['val_loss']))['epoch']
        for split, color, dash in (('train', colors[kind], True), ('val', colors[kind], False)):
            series = [(int(row['epoch']), float(row[f'{split}_loss'])) for row in rows]
            coords = [(x(epoch), y(value)) for epoch, value in series]
            for start, end in zip(coords, coords[1:]):
                if dash:
                    for fraction in [i / 20 for i in range(0, 20, 2)]:
                        f2 = min(fraction + .06, 1)
                        draw.line((start[0] + (end[0]-start[0])*fraction,
                                   start[1] + (end[1]-start[1])*fraction,
                                   start[0] + (end[0]-start[0])*f2,
                                   start[1] + (end[1]-start[1])*f2), fill=color, width=3)
                else:
                    draw.line((start, end), fill=color, width=7)
            for epoch, value in series:
                xx, yy = x(epoch), y(value)
                radius = 7 if split == 'train' else 10
                draw.ellipse((xx-radius, yy-radius, xx+radius, yy+radius),
                             fill='white' if split == 'train' else color,
                             outline=color, width=3)
                if split == 'val' and str(epoch) == str(best_epoch):
                    draw.ellipse((xx-13, yy-13, xx+13, yy+13), outline='#111827', width=2)

    # Separate the two visual encodings in the legend: color identifies model,
    # while line weight/style and marker fill identify train versus validation.
    draw.text((185, 91), 'Model / color', fill='#374151', font=label_font)
    for index, (kind, color) in enumerate(colors.items()):
        bx = 390 + index * 245
        draw.line((bx, 103, bx + 48, 103), fill=color, width=6)
        draw.text((bx + 62, 90), kind.upper(), fill='#111827', font=label_font)

    draw.text((185, 140), 'Split / style', fill='#374151', font=label_font)
    # Validation: solid, heavier line and filled marker.
    draw.line((390, 154, 438, 154), fill='#111827', width=7)
    draw.ellipse((406, 145, 422, 161), fill='#111827', outline='#111827', width=2)
    draw.text((450, 140), 'Validation — solid, bold, filled marker', fill='#111827', font=label_font)
    # Training: dashed, lighter line and hollow marker.
    bx, by = 975, 154
    for start in range(bx, bx + 49, 16):
        draw.line((start, by, min(start + 9, bx + 49), by), fill='#111827', width=4)
    draw.ellipse((bx + 16, by - 7, bx + 30, by + 7), fill='white', outline='#111827', width=2)
    draw.text((1035, 140), 'Training — dashed, hollow marker', fill='#111827', font=label_font)
    # A black ring around a validation point marks the checkpoint retained
    # for that architecture: the epoch with the lowest validation loss.
    draw.ellipse((390, 177, 408, 195), fill='white', outline='#111827', width=2)
    draw.ellipse((395, 182, 403, 190), fill='#10b981', outline='#10b981', width=1)
    draw.text((425, 174), 'Black ring = checkpoint at minimum validation loss', fill='#111827', font=label_font)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format='PNG', dpi=(180, 180))
    print(path)


if __name__ == '__main__':
    main()
