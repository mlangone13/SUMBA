#%matplotlib inline
#%matplotlib notebook


import cv2
import pandas as pd
import numpy as np
import random as rng
from PIL import Image

##########################################################################################
##########################################################################################


class GraspDetection:
    def __init__(self, show, N, tol):
        self.show = show
        self.N = N
        self.tol = tol

    def find_grasping_point(self, src_gray, first_box):
        # Step 5: Get mass center of the object
        draw, _, contour, center = self.get_mass_center(src_gray)

        if self.show:
            img = Image.fromarray(draw)
            img.show()

        # Step 6: Calculate all lines from the mass center
        lines = self.calculate_lines(
            draw, contour, center[0], center[1], self.N, self.tol
        )

        # Step 7: Compute the grasping values
        df = self.compute_grapsing_values(lines, contour, center)

        # if self.show:
        #     print(df.head())

        # Step 8: Determine best grasp
        best_grasp = self.get_best_grasping(draw, df, display=self.show, by="diameter")

        if self.show:
            print("Original value:", first_box)
            print("Best Grasping value:", best_grasp)

        # Step 9: Convert point to absolut points of the image
        best_absolut_grasp = self.parse_absolute_grasp(draw, first_box, best_grasp)

        if self.show:
            print("Final absolut grasp:", best_absolut_grasp)
        return best_absolut_grasp

    def get_mass_center(self, src_gray):

        contours, _ = cv2.findContours(src_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar el centro de masa del objeto
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M["m10"] / (M["m00"] + 1e-5))
        cy = int(M["m01"] / (M["m00"] + 1e-5))

        # Draw contours
        drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)

        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, [c], 0, color, 1)
        cv2.circle(drawing, (int(cx), int(cy)), 1, color, -1)
        return drawing, src_gray, c, (cx, cy)

    def calculate_lines(self, img, cnt, centroid_x, centroid_y, N=60, tol=5):

        # Step #3
        out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_bw = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        # Step #4
        ref = np.zeros_like(img_bw)
        cv2.drawContours(ref, [cnt], 0, 255, 1)

        # Get dimensions of the image
        width = np.array(out).shape[1]
        height = np.array(out).shape[0]

        # Step #6

        final_results = []
        for i in range(N):
            # Step #6a
            tmp = np.zeros_like(img_bw)

            # Step #6b
            theta = i * (360 / N)
            theta2 = theta + 180

            theta2 *= np.pi / 180.0
            theta *= np.pi / 180.0

            cv2.line(
                tmp,
                (
                    int(centroid_x + np.cos(theta2) * width),
                    int(centroid_y - np.sin(theta2) * height),
                ),
                (
                    int(centroid_x + np.cos(theta) * width),
                    int(centroid_y - np.sin(theta) * height),
                ),
                255,
                2,
            )

            # Step #6d
            row, col = np.nonzero(np.logical_and(tmp, ref))

            # Step #6e
            try:
                color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                cv2.line(
                    out,
                    (centroid_x, centroid_y),
                    (col[0], row[0]),
                    color,
                )

                cv2.circle(out, (int(col[0]), int(row[0])), 1, (255, 255, 255), -1)

                idx = 0
                for i in range(len(col)):
                    if abs(col[i] - col[0]) > tol or abs(row[i] - row[0]) > tol:
                        idx = i
                        break

                cv2.circle(out, (int(col[idx]), int(row[idx])), 1, (255, 255, 255), -1)
            except Exception as e:
                print(e)

            final_results.append((col[0], row[0], col[idx], row[idx]))

        return final_results

    def draw_best_points(self, img, x0, y0, x1, y1):
        out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.circle(out, (x0, y0), 1, (255, 255, 255), -1)
        cv2.circle(out, (x1, y1), 1, (255, 255, 255), -1)
        img_end = Image.fromarray(out)
        img_end.show()

    def compute_grapsing_values(self, lines, contour, center):

        filled_contour_area = cv2.contourArea(contour)

        df = pd.DataFrame(lines, columns=["col0", "row0", "col1", "row1"])

        df["center_x"] = center[0]
        df["center_y"] = center[1]

        df["distance_upper_half"] = (
            abs(df["col0"] - df["center_x"]) ** 2
            + abs(df["row0"] - df["center_y"]) ** 2
        ) ** 0.5
        df["distance_lower_half"] = (
            abs(df["col1"] - df["center_x"]) ** 2
            + abs(df["row1"] - df["center_y"]) ** 2
        ) ** 0.5
        df["diff_distances"] = abs(
            df["distance_lower_half"] - df["distance_upper_half"]
        )
        df["diameter"] = abs(df["distance_lower_half"] + df["distance_upper_half"])
        df["full_mass"] = filled_contour_area
        return df

    def get_best_grasping(self, draw, df, display=True, by="diff_distance"):
        sorted_distance = df.sort_values(by=by)
        points = list(sorted_distance.head(1).values[0][0:4])
        points = [int(x) for x in points]
        if display:
            self.draw_best_points(draw, points[0], points[1], points[2], points[3])
        return points

    def parse_absolute_grasp(self, draw, first_box, best_grasp):
        height = draw.shape[0]
        width = draw.shape[1]
        dw = (first_box[2] - first_box[0]) / width
        dh = (first_box[3] - first_box[1]) / height

        best_absolut_grasp = (
            first_box[0] + best_grasp[0] * dw,
            first_box[1] + best_grasp[1] * dh,
            first_box[0] + best_grasp[2] * dw,
            first_box[1] + best_grasp[3] * dh,
        )
        return best_absolut_grasp
