#ifndef LINE_ITERATOR_H
#define LINE_ITERATOR_H
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>

// glue to allow iterating over a filestream almost as easily as Python
// pulled from http://dev-perspective.blogspot.com/2012/09/iterating-over-text-file.html

struct line_t : std::string {
     friend std::istream & operator>>(std::istream& is, line_t& line) {
         std::getline(is,line);
         line_t tmp = line;
         if (line.back() == '\\')
           tmp.pop_back();
         while(line.back() == '\\')
         {
           line.pop_back();
           std::getline(is, line);
           tmp += line;
           if (line.back() == '\\')
             tmp.pop_back();
         }
         line = tmp;
         return is;
     }
};
 
typedef std::istream_iterator<line_t> line_iterator;
 
template <typename T>
class line_range {
    T istr;
    line_iterator b;
 
public:
    typedef line_iterator iterator;
    typedef line_iterator const_iterator;
 
    line_range(T&& is) :
        istr(std::forward<T>(is)),
        b(istr)
    {}
 
    line_iterator begin() const { return b; }
    line_iterator end() const { return line_iterator(); }
};
 
template <typename S>
auto lines(S&& is) -> decltype(line_range<S>(std::forward<S>(is))) {
    return line_range<S>(std::forward<S>(is));
}
#endif
