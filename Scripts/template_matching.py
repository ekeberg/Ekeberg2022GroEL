import numpy
import scipy.optimize
import time

class Matcher:
    def __init__(self, templates, mask):
        self.templates = templates
        self.mask = mask.copy()
        # self.templates *= self.mask[numpy.newaxis, :, :]

    def match(self, pattern):
        dprod = (pattern*(self.templates*self.mask[numpy.newaxis, :, :])).sum(axis=(1, 2))
        return dprod

    
class MatcherBG(Matcher):
    def __init__(self, templates, mask, background, quiet=False):
        super().__init__(templates, mask)
        self.quiet = quiet
        self.background = background.copy()
        self.background *= self.mask

    def match(self, pattern):
        best_fit = 1e100
        best_s = 0
        best_b = 0
        best_index = 0
        all_fit = numpy.zeros(len(self.templates))
        all_s = numpy.zeros(len(self.templates))
        all_b = numpy.zeros(len(self.templates))
        for tindex in range(len(self.templates)):
            if not self.quiet and tindex % 500 == 0:
                print(tindex)
            def func(p):
                s = p[0]; b = p[1]
                return ((s*self.templates[tindex] + b*self.background) - pattern)[self.mask]
            p1, success = scipy.optimize.leastsq(func, [1, 1], epsfcn=1.)
            s = p1[0]; b = p1[1]
            fit = numpy.sqrt((func(p1)**2).sum())
            all_fit[tindex] = fit
            all_s[tindex] = s
            all_b[tindex] = b

        best_index = all_fit.argmin()
        
        return {"fit": all_fit[best_index],
                "index": best_index,
                "signal": all_s[best_index],
                "background": all_b[best_index]}


class MatcherBGCenter(Matcher):
    def __init__(self, templates, mask, background, center_shift, quiet=False):
        super().__init__(templates, mask)
        self.quiet = quiet
        self.background = background.copy()
        self.background *= self.mask
        self.shift_x = center_shift
        self.shift_y = center_shift

    def match(self, pattern):
        best_fit = 1e100
        best_s = None
        best_b = None
        best_shift_x = None
        best_shift_y = None
        best_index = None
        
        for shift_x in self.shift_x:
            start_time = time.time()
            for shift_y in self.shift_y:
                    
                pattern_shifted = numpy.roll(pattern, (shift_x, shift_y), axis=(0, 1))
                mask_shifted = numpy.roll(self.mask, (shift_x, shift_y), axis=(0, 1))
                background_shifted = numpy.roll(self.background, (shift_x, shift_y), axis=(0, 1))
                for tindex in range(len(self.templates)):
                    def func(p):
                        s = p[0]; b = p[1]
                        return ((s*self.templates[tindex] + b*background_shifted) - pattern_shifted)[mask_shifted]
                    p1, success = scipy.optimize.leastsq(func, [1, 1], epsfcn=1.)
                    s = p1[0]; b = p1[1]
                    fit = numpy.sqrt((func(p1)**2).sum())
                    if fit < best_fit:
                        best_fit = fit
                        best_index = tindex
                        best_shift_x = shift_x
                        best_shift_y = shift_y
                        best_s = s
                        best_b = b
                        
            print(f"{shift_x}: took {time.time() - start_time} s", flush=True)
        return {"fit": best_fit, "signal": best_s, "background": best_b, "index": best_index,
                "shift_x": best_shift_x, "shift_y": best_shift_y}
