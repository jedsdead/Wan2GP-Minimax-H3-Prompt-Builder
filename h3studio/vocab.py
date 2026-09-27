"""Vocabularies for the structured builder.

Lists, labels and H3 window figures taken from the user's own
"MiniMax H3 Prompt Builder" plugin (JedsDeadBaby), so the two tools speak the
same language. Kept verbatim so a phrase chosen here reads the way it does there.
"""

STYLES = [
    "",
    # live action, general
    "Live-action, cinematic", "Live-action, documentary",
    "Observational documentary", "Handheld, naturalistic",
    "Multi-camera sitcom", "Music-video", "Vintage film",
    "Home-video footage", "Found-footage", "Mockumentary",
    "Security-camera footage", "Newsreel",
    # live action, genre
    "Neo-noir", "Film noir", "Western", "Spaghetti western",
    "War epic", "Historical epic", "Period drama", "Courtroom drama",
    "Heist thriller", "Spy thriller", "Psychological thriller",
    "Slasher horror", "Cosmic horror", "Body horror", "Gothic horror",
    "Science-fiction epic", "Cyberpunk", "Steampunk", "Post-apocalyptic",
    "Dystopian", "Space opera", "Creature feature", "Disaster movie",
    "Superhero blockbuster", "Action blockbuster", "Martial-arts film",
    "Romantic comedy", "Screwball comedy", "Coming-of-age drama",
    "Road movie", "Sports drama", "Biopic", "Musical", "Fantasy epic",
    "French New Wave", "Italian neorealism", "Kitchen-sink realism",
    "German expressionism", "Silent film", "Technicolor melodrama",
    "1970s New Hollywood", "1980s VHS aesthetic", "Surrealist film",
    # animation
    "2D-animated", "Hand-painted anime film", "Cel-shaded anime",
    "Modern 3D animated feature", "1990s 2D animated feature",
    "1930s rubber-hose cartoon", "Saturday-morning cartoon",
    "Adult animated comedy", "Stop-motion", "Claymation",
    "Puppet animation", "Papercraft animation", "Rotoscoped animation",
    "Motion comic", "Pixel-art animation", "Silhouette animation",
    # illustrative and rendered
    "Watercolour", "Oil-painted", "Charcoal sketch", "Ink and wash",
    "Comic-book panel", "Graphic-novel", "Storyboard sketch",
    "3D CG", "Photorealistic render", "Low-poly render", "Wireframe render",
]

FRAMINGS = [
    "", "extreme close-up", "close-up", "medium close-up", "medium",
    "medium-wide", "wide", "extreme wide", "establishing",
    "over-the-shoulder", "low-angle", "high-angle", "overhead", "worm's-eye",
    "dutch-angle", "point-of-view", "profile", "silhouette", "insert",
    "cutaway", "master", "reflection", "through-the-window",
]

MOTION_TYPES = [
    "", "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
    "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
    "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
    "Roll Counterclockwise",
    # --- beyond the guide's table ---
    "Full 360 Orbit", "Spiral Around Subject", "Dolly Zoom",
    "Crash Zoom In", "Snap Zoom Out", "Whip Pan", "Rack Focus",
    "Trail Behind Subject", "Lead Subject Backwards", "Locked Off",
]

MOTION_VERBS = {
    "Zoom In": "zooms in", "Zoom Out": "zooms out",
    "Push In": "pushes in", "Pull Out": "pulls out",
    "Pan Left": "pans left", "Pan Right": "pans right",
    "Truck Left": "trucks left", "Truck Right": "trucks right",
    "Tilt Up": "tilts up", "Tilt Down": "tilts down",
    "Pedestal Up": "rises on the pedestal",
    "Pedestal Down": "lowers on the pedestal",
    "Arc Shot": "moves in an arc around the subject",
    "Tracking Shot": "tracks the subject",
    "Static Shot": "holds a static shot",
    "Shake Slightly": "shakes slightly", "Shake Strongly": "shakes strongly",
    "POV": "takes the subject's point of view",
    "Roll Clockwise": "rolls clockwise",
    "Roll Counterclockwise": "rolls counterclockwise",
    # --- beyond the guide's table ---
    "Full 360 Orbit": "makes a full 360-degree orbit around the subject",
    "Spiral Around Subject": "spirals around the subject",
    "Dolly Zoom": "performs a dolly zoom",
    "Crash Zoom In": "crash zooms in", "Snap Zoom Out": "snap zooms out",
    "Whip Pan": "whip pans", "Rack Focus": "racks focus",
    "Trail Behind Subject": "trails behind the subject",
    "Lead Subject Backwards": "leads the subject backwards",
    "Locked Off": "holds a locked-off frame",
}

RIGS = [
    "", "tripod", "handheld", "shoulder-rig", "steadicam", "gimbal", "dolly",
    "crane", "jib", "drone", "cable-cam", "car-mounted", "hood-mounted",
    "slider", "motion-control", "underwater", "body-mounted",
    "vehicle-mounted", "rickshaw",
]

AMPLITUDES = ["", "with small amplitude", "with large amplitude"]

SPEEDS = ["", "at slow speed", "at fast speed"]

CUT_VERBS = [
    "the camera cuts to", "the shot cuts to", "the shot hard-cuts to",
    "the shot smash-cuts to", "the shot match-cuts to",
    "the shot jump-cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
    "the shot cross-dissolves to", "the shot dissolves to",
    "the shot fades to", "the shot wipes to", "the shot whip-pans to",
    "the shot irises to", "the shot cuts away to",
]

CONTINUE_VERBS = [
    "the camera moves to",
    "the camera continues into",
    "the camera reframes to",
    "the camera settles into",
    "the camera drifts to",
    "the camera swings round to",
]

LANGUAGES = [
    "English", "Mandarin Chinese", "Cantonese", "Japanese", "Korean",
    "Spanish", "French", "German", "Italian", "Portuguese", "Russian",
    "Polish", "Greek", "Turkish", "Arabic", "Hebrew", "Hindi", "Bengali",
    "Thai", "Vietnamese", "Indonesian", "Tagalog", "Swahili",
    "Swedish", "Norwegian", "Danish", "Finnish", "Irish", "Welsh",
]

VOICE_AGES = ["", "child", "teenage", "young", "young adult", "middle-aged",
              "older", "elderly", "ageless"]

VOICE_GENDERS = ["", "female", "male", "androgynous"]

VOICE_PITCH = ["", "low", "medium", "high"]

VOICE_TIMBRE = [
    "", "clear", "raspy", "breathy", "warm", "nasal", "gravelly", "bright",
    "weathered", "smooth", "resonant", "husky", "rich", "booming",
]

VOICE_RATE = [
    "", "slow", "measured", "unhurried", "quick", "clipped", "halting",
    "breathless", "drawling", "steady", "urgent",
]

CHAR_ETHNICITIES = [
    "", "Asian", "East Asian", "South Asian", "Black", "White",
    "Hispanic", "Middle Eastern", "Native American", "Pacific Islander",
    "Mixed ethnicity",
]

CHAR_GENDERS = ["", "male", "female", "non-binary", "androgynous"]

CHAR_AGE_RANGES = [
    "", "childhood", "the teens", "early 20s", "late 20s", "early 30s",
    "mid-30s", "early 40s", "mid-40s", "the 50s", "the 60s", "the 70s",
    "old age",
]

CHAR_HEIGHTS = [
    "", "short", "petite", "average height", "five foot four",
    "five foot eight", "six feet", "six foot two", "tall",
]

CHAR_BUILDS = [
    "", "slender", "athletic", "toned", "muscular", "stocky", "heavyset",
    "broad-shouldered", "wiry", "curvy", "average build", "petite build",
]

CHAR_HAIRSTYLES = [
    "", "long straight", "long wavy", "long curly", "shoulder-length",
    "short cropped", "buzz cut", "shaved head", "braided", "dreadlocked",
    "afro", "slicked-back", "messy tousled", "ponytailed",
]

CHAR_HAIR_COLORS = [
    "", "black", "dark brown", "light brown", "blonde", "auburn", "red",
    "grey", "white", "salt-and-pepper", "dyed vibrant",
]

CHAR_EYE_COLORS = [
    "", "brown", "dark brown", "blue", "light blue", "green", "hazel",
    "grey", "amber", "black",
]

CHAR_CLOTHING = [
    "", "a plain white t-shirt and jeans", "a rumpled trenchcoat",
    "a tailored black suit", "a floral summer dress", "a long evening gown",
    "a hooded sweatshirt and joggers", "a leather biker jacket",
    "a wool overcoat and scarf", "a white lab coat over scrubs",
    "a stained apron over a work shirt",
    "a high-visibility jacket and work boots", "a police uniform",
    "a military field uniform", "a school uniform",
    "worn workwear, patched at the knees", "traditional formal dress",
]

GRADING = [
    "", "a vibrant colour grade", "a muted colour grade",
    "a high-contrast grade", "a low-contrast grade",
    "a teal-and-orange grade", "a bleach-bypass grade",
    "a warm vintage film grade", "a cool desaturated grade",
    "a pastel low-contrast grade", "a faded retro grade",
    "a rich saturated technicolor grade", "a sepia-toned grade",
    "a cross-processed grade", "a neon cyberpunk grade",
    "a sun-bleached grade", "a cold blue-grey grade",
    "a moody green-tinted grade", "a monochrome grade",
    "a high-key bright grade", "a crushed-blacks grade",
    "an amber golden grade", "a silver-halide grade",
]

VISUAL_RETENTION = ["fully_preserved", "partially_preserved",
                    "attribute_transfer", "weak_reference"]

AUDIO_RETENTION = ["fully_copy", "partially_copy", "reference", "weak_reference"]

TASK_TYPES = ["keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference"]

LOCATIONS = [
    "a rain-slicked alley behind a nightclub", "a farmhouse kitchen",
    "an airport departure lounge", "a lighthouse gallery in a storm",
    "a disused swimming pool", "a rooftop garden above the city",
    "an antique shop crowded with clocks", "a motorway service station",
    "a boat deck on open water", "a stone chapel lit by candles",
    "a launderette", "a records archive in a basement",
    "a ski lift above treeline", "a bustling New York street",
    "a covered market hall", "a tiled underpass",
    "a rain-soaked city street", "a suburban kitchen", "a hotel corridor",
    "a crowded subway platform", "a quiet library reading room",
    "an empty car park", "a coastal fishing dock",
    "a pine forest clearing", "a desert highway", "a rooftop",
    "a hospital waiting room", "a school classroom", "a dive bar",
    "an office", "a country lane", "a snowbound cabin",
    "a train carriage", "a cathedral interior", "a warehouse floor",
    "a greenhouse", "a mountain ridge", "a riverbank",
]

TIMES_OF_DAY = [
    "", "at dawn", "in the early morning", "in the morning", "at midday",
    "in the afternoon", "in the late afternoon", "at golden hour",
    "at sunset", "at dusk", "in the evening", "at night", "late at night",
    "at midnight", "in the small hours",
]

SCENE_LIGHTING = [
    "a single bare bulb", "car headlights sweeping across",
    "television glow in a dark room", "shafts of light through blinds",
    "the cold blue of a screen", "stage lighting from above",
    "a red safelight", "lightning flashes", "sodium streetlight",
    "sunrise backlight through haze",
    "golden hour light", "blue hour light", "harsh midday sun",
    "soft overcast light", "moody low-key lighting", "high-key lighting",
    "flickering candlelight", "neon glow", "a backlit silhouette",
    "practical lamps only", "dramatic hard shadows", "rim lighting",
    "dappled sunlight through leaves", "harsh fluorescent light",
    "moonlight", "firelight", "diffused window light",
    "strobing club lighting", "underwater caustics", "streetlight sodium glow",
]

SCENE_ATMOSPHERE = [
    "torrential rain", "sea spray", "ash falling", "petals drifting",
    "insects circling a lamp", "condensation on every surface",
    "a low ground mist", "sunbeams cutting through dust",
    "thick fog", "light mist", "drizzling rain", "heavy rain",
    "drifting dust", "smoke haze", "floating particles", "still, calm air",
    "gusting wind", "humid haze", "crisp clear air", "falling snow",
    "swirling sand", "rising steam", "heat shimmer", "drifting embers",
    "industrial smog", "morning dew",
]

CAMERA_TYPES = [
    "an IMAX camera", "IMAX 70mm film", "an Arri Alexa digital cinema camera",
    "a RED digital cinema camera", "a Panavision camera",
    "a Blackmagic cinema camera", "Super 35mm film", "Super 16mm film",
    "16mm film", "8mm film", "35mm film", "a 4K digital camera",
    "an 8K digital camera", "a DSLR", "a mirrorless camera",
    "a GoPro action camera", "a drone camera", "a vintage VHS camcorder",
    "a phone camera", "a security camera", "a Polaroid instant camera",
    "black-and-white film stock", "expired film stock",
]

LENS_TYPES = [
    "a 14mm ultra-wide lens", "a 24mm wide-angle lens", "a 35mm lens",
    "a 50mm standard lens", "an 85mm portrait lens", "a 100mm macro lens",
    "a 135mm telephoto lens", "a 200mm telephoto lens", "an anamorphic lens",
    "a fisheye lens", "a tilt-shift lens", "a vintage soft-focus lens",
    "a prime lens", "a zoom lens",
    "a large-format lens with shallow depth of field",
    "a probe lens", "a split-diopter lens",
]

SOUNDSCAPE_PRESETS = [
    "wind across an open field", "rain on a tin roof",
    "a kettle coming to the boil", "a fridge humming in a quiet kitchen",
    "seagulls over a harbour", "an aircraft passing overhead",
    "keyboard tapping and chair creaks", "a crowd in a stadium",
    "hooves on cobblestones", "a clock chiming the hour",
    "surf dragging over shingle", "a generator running outside",
    "room tone and distant traffic", "rain against windows",
    "wind through trees", "birdsong and rustling leaves",
    "waves against a shoreline", "a crackling fire",
    "crowd chatter and clinking glasses", "market noise and trolley wheels",
    "fluorescent hum and footsteps on tile", "machinery hum",
    "a ticking clock in a quiet room", "distant sirens",
    "train rumble through a tunnel", "church bells",
    "cicadas in the heat", "creaking timber", "coins and paper shifting",
    "breathing and shifting fabric", "footsteps on gravel",
    "a washing drum tumbling", "near silence with faint air movement",
]

MUSIC_INSTRUMENTS = [
    "", "a restrained solo piano",
    "sustained low strings",
    "a lone cello line",
    "a full string section",
    "brushed drums and upright bass",
    "a muted trumpet over brushed drums",
    "fingerpicked acoustic guitar",
    "electric guitar with long reverb",
    "warm analogue synth pads",
    "a pulsing synth arpeggio",
    "harp and woodwinds",
    "low brass and timpani",
    "a heroic brass fanfare",
    "sparse percussion and hand claps",
]

MUSIC_TEMPOS = [
    "", "at a slow tempo", "at a moderate tempo", "at a brisk tempo",
    "at a driving tempo", "on a steady pulse", "rubato, with no fixed pulse",
]

MUSIC_DYNAMICS = [
    "", "held quietly under the scene",
    "swelling gradually, then falling away",
    "building steadily to a peak",
    "entering late and fading at the end",
    "with no swell",
    "dropping out abruptly at the cut",
]

ASSET_KINDS = ["Subject", "Picture", "Video", "Audio"]

REF_VIDEO_SLOTS = ["", "Video 1", "Video 2"]

REF_AUDIO_SLOTS = ["", "Audio 1", "Audio 2"]

REF_PICTURE_SLOTS = [f"Picture {n + 1}" for n in range(9)]

H3_FPS = 24

H3_WINDOW_FRAMES_MIN = 107            # 4.46s - one window cannot be shorter

H3_WINDOW_FRAMES_MAX = 481            # 20.04s - the Sliding Window Size cap

H3_WINDOW_DOC_SECONDS = (4.0, 15.0)   # the documented band

H3_OVERLAP_DEFAULT = 18               # overlap_default; the ladder is 17k+1

