import firstPython as fp


def test_success1():
	assert fp.addX(3)==4

def test_success2():
	assert fp.addX(-3)==-2

def test_failure1():
	assert fp.addX(3)==5

def test_failure2():
	assert fp.addX(-3)==4
