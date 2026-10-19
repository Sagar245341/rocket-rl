"""Procedural rocket and coastal launchpad artwork; no external assets."""
import math
import pygame


def draw_world(env, size=(900, 590)):
    w, h = size
    surface = pygame.Surface(size)
    horizon = h - 105
    for y in range(h):
        t = y / h
        pygame.draw.line(surface, (int(7 + 16*t), int(14 + 24*t), int(31 + 25*t)), (0, y), (w, y))
    for i in range(85):
        x, y = (i * 137 + 43) % w, (i * 79 + 11) % (horizon - 70)
        pygame.draw.circle(surface, (100 + i % 90, 140 + i % 70, 190), (x, y), 1)
    pygame.draw.circle(surface, (43, 61, 83), (w - 110, 80), 28)
    pygame.draw.circle(surface, (170, 187, 199), (w - 115, 75), 22)
    pygame.draw.polygon(surface, (23, 41, 56), [(0,horizon), (0,horizon-25), (130,horizon-65),
                        (240,horizon-15), (420,horizon-40), (620,horizon-5), (w,horizon-45), (w,horizon)])
    pygame.draw.rect(surface, (13, 38, 53), (0, horizon, w, h-horizon))
    for i in range(18):
        y = horizon + 8 + i * 6
        start = (i * 97) % w
        pygame.draw.line(surface, (25, 62, 76), (start, y), (min(w, start + 120), y))
    ground = h - 76
    scale_x = (w - 90) / 310
    scale_y = (ground - 55) / 260

    def world(x, y):
        return round(w / 2 + x * scale_x), round(ground - y * scale_y)

    # Industrial landing platform, floodlights, hazard stripes and illuminated target.
    px, py = world(0, 0)
    half = round(env.pad_half_width * scale_x)
    pygame.draw.polygon(surface, (37, 52, 66), [(px-100,py+22),(px+100,py+22),(px+135,h),(px-135,h)])
    pygame.draw.rect(surface, (82, 102, 113), (px-half-12, py, half*2+24, 12), border_radius=3)
    pygame.draw.rect(surface, (43, 66, 75), (px-half, py-4, half*2, 5))
    for x in range(px-half, px+half, 12):
        pygame.draw.line(surface, (226, 174, 68), (x,py+8), (x+7,py+2), 3)
    pygame.draw.ellipse(surface, (65, 235, 199), (px-half+5, py-7, half*2-10, 9), 2)
    pygame.draw.line(surface, (205, 255, 240), (px-8,py-2), (px+8,py-2), 2)
    for x in [px-half-20, px+half+20]:
        pygame.draw.line(surface, (88, 117, 135), (x,py+5), (x,py-32), 3)
        pygame.draw.circle(surface, (83, 221, 205), (x,py-32), 4)
    # Range tower and small service structures.
    tx = 100
    pygame.draw.rect(surface, (31, 49, 65), (tx,ground-105,22,110))
    for y in range(ground-100, ground, 18):
        pygame.draw.line(surface, (65, 87, 104), (tx,y), (tx+22,y+18), 2)
        pygame.draw.line(surface, (65, 87, 104), (tx+22,y), (tx,y+18), 2)
    pygame.draw.line(surface, (98, 119, 138), (tx+11,ground-105), (tx+11,ground-145), 2)
    pygame.draw.circle(surface, (255, 116, 105), (tx+11,ground-145), 3)
    for x,y in env.trajectory[-240:]:
        pygame.draw.circle(surface, (53, 105, 128), world(x,y), 1)
    rx, foot_y = world(env.x, env.y)
    # Rocket has a fixed readable display scale; physics uses foot clearance.
    rocket = pygame.Surface((100, 150), pygame.SRCALPHA)
    if env.throttle > 0 and env.fuel >= 0 and env.outcome == 'FLYING':
        length = 12 + int(38 * env.throttle) + int(4 * math.sin(env.steps * 1.7))
        pygame.draw.polygon(rocket, (255, 105, 40, 100), [(39,101),(61,101),(50,110+length)])
        pygame.draw.polygon(rocket, (255, 190, 62), [(43,102),(57,102),(50,107+length)])
        pygame.draw.polygon(rocket, (222, 249, 255), [(46,102),(54,102),(50,112+length//2)])
    pygame.draw.polygon(rocket, (200, 216, 226), [(40,30),(50,12),(60,30)])
    pygame.draw.rect(rocket, (224, 231, 234), (40,30,20,66))
    pygame.draw.rect(rocket, (156, 178, 193), (40,30,5,66))
    pygame.draw.rect(rocket, (247, 249, 245), (54,31,4,63))
    pygame.draw.rect(rocket, (28, 43, 57), (40,39,20,14))
    pygame.draw.rect(rocket, (43, 60, 73), (40,84,20,10))
    for y in [57, 78]:
        pygame.draw.line(rocket, (120, 146, 165), (41,y), (59,y))
    pygame.draw.polygon(rocket, (115, 141, 157), [(43,94),(57,94),(61,103),(39,103)])
    for side in [-1,1]:
        pygame.draw.line(rocket, (168, 188, 198), (50+side*8,85), (50+side*24,111), 3)
        pygame.draw.line(rocket, (120, 147, 166), (50+side*8,96), (50+side*24,111), 2)
        pygame.draw.line(rocket, (212, 225, 226), (50+side*27,112), (50+side*20,112), 3)
        pygame.draw.rect(rocket, (82, 106, 121), (50+side*15-3,56,6,12))
    pygame.draw.rect(rocket, (45, 178, 177), (47,62,6,9))
    rotated = pygame.transform.rotate(rocket, -math.degrees(env.angle))
    surface.blit(rotated, rotated.get_rect(center=(rx, foot_y-37)))
    if env.outcome == 'CRASHED':
        pygame.draw.circle(surface, (249, 129, 65), (rx,foot_y-8), 28, 3)
        for i in range(8):
            a = i * math.pi / 4
            pygame.draw.line(surface, (255,179,72), (rx+int(30*math.cos(a)),foot_y+int(30*math.sin(a))),
                             (rx+int(44*math.cos(a)),foot_y+int(44*math.sin(a))), 2)
    return surface
