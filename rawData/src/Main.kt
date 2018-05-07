import java.io.BufferedWriter
import java.io.File
import java.io.InputStream
import java.io.OutputStream

fun main(args: Array<String>) {
	val rawDataDir = File("./rawData")
	val trainDataDir = File("./trainData").apply { mkdirs() }

	val trainDataFile = trainDataDir.resolve("./trainData.txt")
	val illegalDataFile = trainDataDir.resolve("./illegalData.txt")

	val legalStream = trainDataFile.outputStream()
	val illegalStream = illegalDataFile.outputStream()

	rawDataDir.listFiles().forEach { rawDataFile ->
		val rawStream = rawDataFile.inputStream()
		handleFile(rawStream, legalStream, illegalStream)
	}
}

fun handleFile(rawStream: InputStream, legalStream: OutputStream, illegalStream: OutputStream) {
	val semiWriter = legalStream.bufferedWriter()
	val illegalWriter = illegalStream.bufferedWriter()
	rawStream.bufferedReader().useLines { lines ->
		val iterator = lines.iterator()
		val block = ArrayList<String>()
		
		while (true) {
			if (!iterator.hasNext()) {
				handleBlock(block, semiWriter, illegalWriter)
				block.clear()
				break
			}
			
			val line = iterator.next()
			if (line.isBlank()) {
				handleBlock(block, semiWriter, illegalWriter)
				block.clear()
				continue
			}
			
			block.add(line)
		}
		
	}
	semiWriter.flush()
	illegalWriter.flush()
}

fun handleBlock(block: List<String>, legalWriter: BufferedWriter, illegalWriter: BufferedWriter) {
	val result = parseBlock(block)
	when (result) {
		is BlockParsed -> {
			legalWriter.write(result.run { "$word $accent" })
			legalWriter.newLine()
		}
		is BlockIllegal -> {
			val builder = StringBuilder()
			builder.appendln(block.joinToString("\n"))
			builder.append("#ERROR:").appendln(result.message).appendln()
			illegalWriter.write(builder.toString())
		}
		is BlockReachToEnd -> return
	}
}

fun parseBlock(lines: List<String>): BlockResult {
	val i = lines.indexOfFirst { it.isNotBlank() }
	val wordLine: String = lines.getOrNull(i)?.trimStart { it == '\uFEFF' }?.trim() ?: return BlockReachToEnd()
	val meaningLine: String = lines.getOrNull(i + 1)?.trim() ?: return BlockIllegal("Unexpectedly reach end before reading the accent!")
	if (meaningLine.isBlank()) return BlockIllegal("A blank line after the word!")
	val sign = meaningLine.split(' ', '.').firstOrNull { it.isNotBlank() }?.toIntOrNull()
		?: return BlockIllegal("There is no signed accent!")
	if (sign == 0) return BlockIllegal("The accent sign must not be 0!")
	val accent = findAccent(wordLine, sign)
	if (accent == -1) return BlockIllegal("Unresolved index of the accent.")
	return BlockParsed(wordLine, accent)
}

sealed class BlockResult
class BlockParsed(val word: String, val accent: Int) : BlockResult()
class BlockIllegal(val message: String) : BlockResult()
class BlockReachToEnd : BlockResult()

const val vowels = "аоуыэяёюие"
fun findAccent(word: String, sign: Int): Int {
	if (sign < 0) return word.length - 1 - findAccent(word.reversed(), -sign)
	
	var count = 0
	word.forEachIndexed { index, char ->
		if (vowels.indexOf(char, ignoreCase = true) != -1) {
			count++
			if (count == sign) return index
		}
	}
	return -1
}
