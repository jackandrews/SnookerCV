import cv2
import os
import numpy as np
import math

class TableEdge():
    def __init__(self, im, x1, y1, x2, y2):
        self.im = im
        self.im2 = cv2.cv.fromarray(self.im)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.edge_pos = "unknown"

        self.d = 3

    def get_offset_coords1(self):
        L = math.sqrt((self.x1-self.x2)*(self.x1-self.x2)+(self.y1-self.y2)*(self.y1-self.y2))

        x1p = int(self.x1 + self.d * (self.y2-self.y1) / L)
        x2p = int(self.x2 + self.d * (self.y2-self.y1) / L)
        y1p = int(self.y1 + self.d * (self.x1-self.x2) / L)
        y2p = int(self.y2 + self.d * (self.x1-self.x2) / L)
        return x1p, y1p, x2p, y2p

    def get_offset_coords2(self):
        L = math.sqrt((self.x1-self.x2)*(self.x1-self.x2)+(self.y1-self.y2)*(self.y1-self.y2))

        x1p = int(self.x1 - self.d * (self.y2-self.y1) / L)
        x2p = int(self.x2 - self.d * (self.y2-self.y1) / L)
        y1p = int(self.y1 - self.d * (self.x1-self.x2) / L)
        y2p = int(self.y2 - self.d * (self.x1-self.x2) / L)
        return x1p, y1p, x2p, y2p

    def get_offset_surface_type(self, x1, y1, x2, y2):
        """
        finds the surface type of line defined by (x1, y1, x2, y2)
        Expects an HSV image input
        """
        line = cv2.cv.InitLineIterator(self.im2, (x1, y1), (x2, y2))
        h_l, s_l, v_l = [], [], []
        for (h, s, v) in line:
            h_l.append(h)
            s_l.append(s)
            v_l.append(v)

        hue = np.median(h_l)
        val = np.median(v_l)

        if hue > 50 and hue < 90:
            return "cloth"
        #elif val < 50:
        #    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        #    return "wood"
        else:
            return "wood"

    def is_edge(self):
        """ This function returns true if the edge has wood on one side of it and cloth on the other """

        # get the surface types either side of edge
        (x1p, y1p, x2p, y2p) = self.get_offset_coords1()
        surface1 = self.get_offset_surface_type(x1p, y1p, x2p, y2p)
        (x1p, y1p, x2p, y2p) = self.get_offset_coords2()
        surface2 = self.get_offset_surface_type(x1p, y1p, x2p, y2p)

        if (surface1 == "wood" and surface2 == "cloth") or (surface2 == "wood" and surface1 == "cloth"):
            self.edge_pos = self.get_edge_pos()
            return True
        return False

    def get_edge_pos(self):
        """
        This returns "top", "bottom, "left" or "right", just by looking at coords of line points
        """

        height, width = self.im.shape[:2]
        if abs(self.y1-self.y2) < 10:   #is it vaguely horizontal?
            if self.y1 < height/2:      #if in the top half of the screen, then top edge
                return "top"
            else:
                return "bottom"

        #if the line moves from left to right in the image, its a left hand side edge (perspective)
        #note we don't know which way round the points are, so check both
        if (self.y1 < self.y2) and (self.x1 > self.x2) or \
           (self.y2 < self.y1) and (self.x2 > self.x1):
            return "left"
        else:
            return "right"

class Table:
    def __init__(self, im):
        self.im = im
        self.im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        edges = self.find_edges()
        edges = self.refine_edges(edges)
        self.corners = self.calc_corners(edges)

    def find_edges(self):
        all_edges = []
        im_grey = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
        im_edges = cv2.Canny(im_grey, 80, 40, apertureSize=3)
        lines = cv2.HoughLinesP(im_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
        for x1, y1, x2, y2 in lines[0]:
            edge = TableEdge(self.im_hsv, x1, y1, x2, y2)
            if edge.is_edge():
                all_edges.append(edge)

        #group sets of edges
        left_edges = [edge for edge in all_edges if edge.edge_pos == "left"]
        right_edges = [edge for edge in all_edges if edge.edge_pos == "right"]
        top_edges = [edge for edge in all_edges if edge.edge_pos == "top"]
        bottom_edges = [edge for edge in all_edges if edge.edge_pos == "bottom"]

        #merge groups of edges
        edges = {}
        edges['left'] = self.merge_edges(left_edges)
        edges['right'] = self.merge_edges(right_edges)
        edges['top'] = self.merge_edges(top_edges)
        edges['bottom'] = self.merge_edges(bottom_edges)

        return edges

    def merge_edges(self, edges):
        points = []
        for edge in edges:
            points.append([edge.x1, edge.y1])
            points.append([edge.x2, edge.y2])

        points = np.reshape(points, (len(edges)*2,2))

        [vx, vy, x, y] = cv2.fitLine(points, cv2.cv.CV_DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((self.im.shape[1]-x)*vy/vx)+y)

        cv2.line(self.im, (self.im.shape[1]-1, righty), (0, lefty), (0, 0, 255), 1)

        return vx, vy, x, y

    def refine_edges(self, edges):
        #TODO: refine accuracy of edge by wiggling around
        return edges

    def calc_corners(self, edges):
        corners = {}
        corners['top_left'] = self.calc_line_intersect(edges['top'], edges['left'])
        corners['top_right'] = self.calc_line_intersect(edges['top'], edges['right'])
        corners['bottom_left'] = self.calc_line_intersect(edges['bottom'], edges['left'])
        corners['bottom_right'] = self.calc_line_intersect(edges['bottom'], edges['right'])

        return corners

    def calc_line_intersect(self, line1, line2):
        vx1, vy1, x1, y1 = line1
        vx2, vy2, x2, y2 = line2

        dx = x2 - x1
        dy = y2 - y1

        det = vx2 * vy1 - vy2 * vx1
        u = (dy * vx2 - dx * vy2) / det

        x = x1 + vx1 * u
        y = y1 + vy1 * u

        return x, y

def ocr_score(im, x1, y1, x2, y2):
    score_a = img[x1:x2, y1:y2]
    cv2.imwrite("score.bmp", score_a)

    os.system("tesseract score.bmp output -psm 8 tess_config")
    os.remove("score.bmp")
    with open("output.txt") as f:
        score = int(f.read())
    os.remove("output.txt")

    return score


img = cv2.imread("table.png")
print ocr_score(img, 441, 340, 460, 364)

table = Table(img)
src = np.array([table.corners['bottom_left'], table.corners['bottom_right'], table.corners['top_right'], table.corners['top_left']])

#the size of a snooker table in mm
dest = np.array([[0, 0], [1778, 0], [1778, 3569], [0, 3569]], np.float32)
dest = dest / 5

transform = cv2.getPerspectiveTransform(src, dest)
im_warp = cv2.warpPerspective(img, transform, (600, 800))

for corner in table.corners:
    x, y = table.corners[corner]
    cv2.circle(img, (x, y), 5, (0, 255, 0))
    cv2.putText(img, corner, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

cv2.imshow('warp', im_warp)

#cv2.imshow('edges', im_edges)
cv2.imshow('blur', img)
#cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





#
#im_blur = cv2.GaussianBlur(im_grey, (0, 0), 1)
#im_edges = cv2.Canny(im_grey, 80, 40, apertureSize = 3)
#circles = cv2.HoughCircles(im_edges,cv2.cv.CV_HOUGH_GRADIENT,1,minDist=10,param1=50,param2=30,minRadius=5,maxRadius=8)
#print circles
#for i in circles[0]:
#    # draw the outer circle
#    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#    # draw the center of the circle
#    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

#lines = cv2.HoughLines(im_edges, rho=1, theta=np.pi/180, threshold=170)
#for rho,theta in lines[0]:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    x0 = a*rho
#    y0 = b*rho
#    x1 = int(x0 + 1000*(-b))
#    y1 = int(y0 + 1000*(a))
#    x2 = int(x0 - 1000*(-b))
#    y2 = int(y0 - 1000*(a))
#    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)