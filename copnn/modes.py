class Mode:
    def __init__(self, mode_par):
        self.mode_par = mode_par
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.mode_par == other
        else:
            return NotImplemented
    
    def __str__(self):
        return self.mode_par
    
    def sample_re(self):
        raise NotImplementedError('The sample_re method is not implemented.')
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')


class Categorical(Mode):
    def __init__(self):
        super().__init__('categorical')
    
    def sample_re(self):
        raise NotImplementedError('The sample_re method is not implemented.')
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')


class Longitudinal(Mode):
    def __init__(self):
        super().__init__('longitudinal')
    
    def sample_re(self):
        raise NotImplementedError('The sample_re method is not implemented.')
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')


class Spatial(Mode):
    def __init__(self):
        super().__init__('spatial')
    
    def sample_re(self):
        raise NotImplementedError('The sample_re method is not implemented.')
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')