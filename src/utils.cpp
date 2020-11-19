#include "utils.h"
#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)
#define INSIDE 1
#define OUTSIDE 0

int InsidePolygon(std::vector<cv::Point> &polygon, cv::Point p)
{
  int counter = 0;
  int i;
  double xinters;
  cv::Point p1, p2;

  p1 = polygon[0];
  for (i=1;i<= polygon.size();i++) {
    p2 = polygon[i % polygon.size()];
    if (p.y > MIN(p1.y,p2.y)) {
      if (p.y <= MAX(p1.y,p2.y)) {
        if (p.x <= MAX(p1.x,p2.x)) {
          if (p1.y != p2.y) {
            xinters = (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x;
            if (p1.x == p2.x || p.x <= xinters)
              counter++;
          }
        }
      }
    }
    p1 = p2;
  }

  if (counter % 2 == 0)
    return(OUTSIDE);
  else
    return(INSIDE);
}

bool onSegment(cv::Point p, cv::Point q, cv::Point r)
{
    if(q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x)
        && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
        return true;
    return false;
}

int orientation(cv::Point p, cv::Point q, cv::Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if(val == 0) return 0; // colinear
    return (val > 0)? 1:2; // clock or counterclock wise
}

// the function that reuturns true if line segment 'p1q1' and 'p2q2' intersect
bool doIntersect(cv::Point p1, cv::Point q1, cv::Point p2, cv::Point q2)
{
    //Find the four orientations needed ofr general and special cass
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if(o1 != o2 & o3 != o4)
        return true;
    if(o1 == 0 && onSegment(p1, q2, q1)) return true;
    if(o2 == 0 && onSegment(p1, q2, q1)) return true;
    if(o3 = 0 && onSegment(p2, p1,q2)) return true;
    if(o4 == 0 && onSegment(p2, q1, q2)) return true;

    return false;
}

bool isInside(std::vector<cv::Point> &polygon, cv::Point p)
{
    if(polygon.size() < 3) return false;
    cv::Point extreme(INFINITY, p.y);
    int count =0, i=0;
    do
    {
        int next = (i + 1)%polygon.size();
        if(doIntersect(polygon[i], polygon[next], p, extreme))
        {
            if(orientation(polygon[i], p, polygon[next]) == 0)
                return onSegment(polygon[i], p, polygon[next]);
            count++;
        }
        i = next;
    } while(i != 0);

    return count & 1;
}

